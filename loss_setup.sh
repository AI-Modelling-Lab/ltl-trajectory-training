# Build and set up the OCaml executables from pregenerated code to the point
# where the case studies can be run (in notebooks from a properly configured
# Python environment).

# Make sure opam is available
if ! command -v opam &> /dev/null
then
    echo "opam could not be found"
    exit 1
fi

# And that it has the required packages
if ! opam list --check -i "ocaml.5*"  &> /dev/null
then
    echo "opam package 'ocaml.5*' was not found (note the version constraint)"
    exit 1
fi
if ! opam list --check -i zarith &> /dev/null
then
    echo "opam package 'zarith' was not found"
    exit 1
fi
if ! opam list --check -i angstrom &> /dev/null
then
    echo "opam package 'angstrom' was not found"
    exit 1
fi

# Make sure dune is available
if ! command -v dune &> /dev/null
then
    echo "dune could not be found"
    exit 1
fi

# Navigate to OCaml folder, build the executable and copy it next to the Python
# scripts
pushd ocaml
dune build
dune install --bindir=$(realpath "execs")
cp execs/ltl_loss ../python/ltl-loss-ocaml
popd
