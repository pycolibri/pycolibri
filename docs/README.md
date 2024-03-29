# how to execute the sphinx documentation 

This documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), a tool that makes it easy to create intelligent and beautiful documentation, for execute the documentation you need to follow the next steps. U need a terminal to execute the commands in this directory.

1. cd to the docs directory
```bash
cd docs
```

2. Install the docs requirements
```bash
pip install -r requirements.txt
```

3. Build the documentation
```bash
make html
```

4. Open the documentation, there is a index.html file in the _build/html directory
```bash
open _build/html/index.html
```

5. Clean the documentation
```bash
make clean
```
