rm -r source/autodoc/*
make clean

#Don't use: 
#  undoc-members (pollutes with garbage internals)
#  imported-members (pollutes with inhereted garbage)

SPHINX_APIDOC_OPTIONS=members,inherited-members,show-inheritance sphinx-apidoc ../forge/ -fMe --implicit-namespaces -o source/autodoc/
SPHINX_APIDOC_OPTIONS=members,inherited-members,show-inheritance sphinx-apidoc ../projekt/ -fMe --implicit-namespaces -o source/autodoc/

#Strip bad headers
for f in source/autodoc/*.rst; do\
   python postprocess.py $f
done

make html

#Working on fixing namespaces
#repren --from "s/^(?:[a-zA-Z0-9]*[.])*([a-zA-Z0-9]+) (package|module)" --to "\1 \2" --full --dry-run source/autodoc/*.rst

