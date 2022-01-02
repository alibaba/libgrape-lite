#Makefile to fetch and install graph data for regression
#testing borrowed from Royal Caliber

#Each graph lives in its own directory
SUBDIRS = ak2010                    \
          arabic-2005               \
          asia_osm                  \
          belgium_osm               \
          cit-Patents               \
          coAuthorsDBLP             \
          delaunay_n13              \
          delaunay_n21              \
          delaunay_n24              \
          europe_osm                \
          germany_osm               \
          hollywood-2009            \
          indochina-2004            \
          kron_g500-logn21          \
          rgg_n_2_24_s0             \
          road_central              \
          road_usa                  \
          roadNet-CA                \
          soc-LiveJournal1          \
          soc-orkut                 \
          soc-sinaweibo             \
          soc-twitter-2010          \
          uk-2002                   \
          uk-2005                   \
          webbase-1M                \
          webbase-2001

.PHONY: $(GRAPHS)

#fetches all graphs, extracts and sets up files for tests
all: recurse

#only download the graphs, but do not proceed further
fetch: recurse

#clean everything except the downloaded graphs
clean: recurse

#clean everything including the downloaded graphs
realclean: recurse

#recurse into each subdirectory and try to build the provided targets
recurse:
	for subdir in $(SUBDIRS); do $(MAKE) -C $$subdir $(MAKECMDGOALS); done

$(GRAPHS):
	$(MAKE) -C $@
