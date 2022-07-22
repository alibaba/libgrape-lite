#common make file fragment for ufl graph datasets
#just define GRAPH_NAME prior to including this fragment

GRAPH_TAR  = $(GRAPH_NAME).tar.gz

setup: $(GRAPH_NAME).mtx

$(GRAPH_NAME).mtx: $(GRAPH_TAR)
	tar xvfz $(GRAPH_TAR) && \
	mv $(GRAPH_NAME)/$(GRAPH_NAME).mtx $(GRAPH_NAME).mtx && \
	rm -rf $(GRAPH_NAME) && \
	../associate_weights.py $(GRAPH_NAME).mtx

clean:
	#rm -f $(GRAPH_NAME).mtx
	rm -f *.mtx

realclean: clean
	#rm -f $(GRAPH_TAR)
	rm -f *.tar.gz

