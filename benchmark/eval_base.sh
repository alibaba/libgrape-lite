#!/usr/bin/env bash

dataset_prefix=/disk4/liang/new_dataset
gunrock_prefix=/disk4/liang/gunrock/build/bin
groute_prefix=/disk4/liang/groute/build
grape_prefix=/disk4/liang/benchmark/libgrape-lite/8d1cba903fcaf054774835ff6624952ca7b8866b
grape_ser_prefix=/disk4/liang/ser
dataset_source_node=('soc-LiveJournal1' '1'
  'soc-twitter-2010' '1'
  'uk-2002' '1'
  'uk-2005' '1'
  'webbase-2001' '1'
  'hollywood-2009' '1'
  'soc-sinaweibo' '1'
  'soc-orkut' '1'
  'indochina-2004' '1'
  'arabic-2005' '1'
  'kron_g500-logn21' '1'
  'delaunay_n24' '1'
  'cit-Patents' '1'
  'road_usa' '1'
  'europe_osm' '1')
