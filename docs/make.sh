rm source/autodoc/*

SPHINX_APIDOC_OPTIONS=members,inherited-members,show-inheritance sphinx-apidoc ../forge/ -fMe --implicit-namespaces -o source/autodoc/

#for f in source/autodoc/*.rst; do\
#   perl -pi -e 's/(module|package)$$// if $$. == 1' $$f ;\
#done

make html

#Working on fixing namespaces
#repren --from "s/^(?:[a-zA-Z0-9]*[.])*([a-zA-Z0-9]+) (package|module)" --to "\1 \2" --full --dry-run source/autodoc/*.rst
