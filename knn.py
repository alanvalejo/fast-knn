#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
kNN (K Nearest Neighbor Graph Construction)
=====================================================

Copyright (C) 2016 Alan Valejo <alanvalejo@gmail.com> All rights reserved
Copyright (C) 2016 Thiago Faleiros <thiagodepaulo@gmail.com> All rights reserved

The nearest neighbor or, in general, the k nearest neighbor (kNN) graph of a
data set is obtained by connecting each instance in the data set to its k
closest instances from the data set, where a distance metric defines closeness.

This file is part of kNN.

kNN is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

kNN is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with kNN. If not, see <http://www.gnu.org/licenses/>.
"""

import csv
import os
import sys
import argparse
import numpy as np

from multiprocessing import Pipe, Process
from helper import write_ncol, write_pajek
from scipy import spatial

__maintainer__ = 'Alan Valejo'
__author__ = 'Alan Valejo, Thiago Faleiros'
__email__ = 'alanvalejo@gmail.com', 'thiagodepaulo@gmail.com'
__credits__ = ['Alan Valejo', 'Thiago Faleiros']
__homepage__ = 'https://github.com/alanvalejo/knn'
__license__ = 'GNU'
__docformat__ = 'markdown en'
__version__ = '0.1'
__date__ = '2016-12-01'

def knn(obj_subset, data, kdtree, k, sender):
	""" K nearest neighbor graph construction.

	Args:
		obj_subset (array): Set of vertices by threads
		data (np.array): Original data table
		kdtree (spatial.KDTree): KD tree accounting for from data
		k (int): K nearest neighbors
		sender (multiprocessing.Connection): Pipe connection objects
	"""

	ew = [] # Set of weighted edges
	for obj in obj_subset:
		obj_attrs = data[obj]
		obj_knn = kdtree.query(obj_attrs, k=(k + 1))
		# For each KNN vertex
		for i, nn in enumerate(obj_knn[1]):
			if obj == nn: continue
			d1 = obj_knn[0][i]
			ew.append((obj, nn, 1 / (1 + d1)))

	sender.send(ew)

def main():
	""" Main entry point for the application when run from the command line. """

	# Parse options command line
	usage = 'use "%(prog)s --help" for more information'
	description = 'kNN graph construction'
	parser = argparse.ArgumentParser(description=description, usage=usage, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=30, width=100))
	optional = parser._action_groups.pop()
	required = parser.add_argument_group('required arguments')
	required.add_argument('-f', '--filename', dest='filename', action='store', type=str, metavar='FILE', default=None, help='name of the %(metavar)s to be loaded')
	optional.add_argument('-d', '--directory', dest='directory', action='store', type=str, metavar='DIR', default=None, help='directory of FILE if it is not current directory')
	optional.add_argument('-o', '--output', dest='output', action='store', type=str, metavar='FILE', default=None, help='name of the %(metavar)s to be save')
	required.add_argument('-l', '--label', dest='label', action='store', type=str, metavar='FILE', default=None, help='list of labels points used to construct RGCLI')
	optional.add_argument('-k', '--k', dest='k', action='store', type=int, metavar='int', default=3, help='kNN (default: %(default)s)')
	optional.add_argument('-t', '--threads', dest='threads', action='store', type=int, metavar='int', default=4, help='number of threads (default: %(default)s)')
	optional.add_argument('-e', '--format', dest='format', action='store', choices=['ncol', 'pajek'], type=str, metavar='str', default='ncol', help='format output file. Allowed values are ' + ', '.join(['ncol', 'pajek']) + ' (default: %(default)s)')
	optional.add_argument('-c', '--skip_last_column', dest='skip_last_column', action='store_false', default=True, help='skip last column (default: true)')
	parser._action_groups.append(optional)
	options = parser.parse_args()

	# Process options and args
	if options.filename is None:
		parser.error('required -f [filename] arg.')
	if options.format not in ['ncol', 'pajek']:
		parser.error('supported formats: ncol and pajek.')
	if options.directory is None:
		options.directory = os.path.dirname(os.path.abspath(options.filename))
	else:
		if not os.path.exists(options.directory): os.makedirs(options.directory)
	if not options.directory.endswith('/'): options.directory += '/'
	if options.output is None:
		filename, extension = os.path.splitext(os.path.basename(options.filename))
		options.output = options.directory + filename + '-knn' + str(options.k) + '.' + options.format
	else:
		options.output = options.directory + options.output

	# Detect wich delimiter and which columns to use is used in the data
	with open(options.filename, 'r') as f:
		first_line = f.readline()
	sniffer = csv.Sniffer()
	dialect = sniffer.sniff(first_line)
	ncols = len(first_line.split(dialect.delimiter))
	if not options.skip_last_column: ncols -= 1

	# Reading data table
	# Acess value by data[object_id][attribute_id]
	# Acess all attributs of an object by data[object_id]
	# To transpose set arg unpack=True
	data = np.loadtxt(options.filename, delimiter=dialect.delimiter, usecols=range(0, ncols))
	attr_count = data.shape[1] # Number of attributes
	obj_count = data.shape[0] # Number of objects
	obj_set = range(0, obj_count) # Set of objects

	# Create KD tree from data
	kdtree = spatial.KDTree(data)

	# Size of the set of vertices by threads, such that V = {V_1, ..., V_{threads} and part = |V_i|
	part = obj_count / options.threads

	# Starting Knn processing
	receivers = []
	for i in xrange(0, obj_count, part):
		sender, receiver = Pipe()
		p = Process(target=knn, args=(obj_set[i:i + part], data, kdtree, options.k, sender))
		p.daemon = True
		p.start()
		receivers.append(receiver)

	# Create set of weighted edges
	edgelist = ''
	for receiver in receivers:
		# Waiting threads
		ew = receiver.recv()
		for edge in ew:
			edgelist += '%s %s %s\n' % edge

	# Save edgelist in output file
	if options.format == 'ncol':
		write_ncol(options.output, edgelist)
	else:
		write_pajek(options.output, obj_count, edgelist)

if __name__ == "__main__":
	sys.exit(main())
