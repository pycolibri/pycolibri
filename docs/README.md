# how to execute the sphinx documentation 

This documentation is built using [Sphinx](https://www.sphinx-doc.org/en/master/), a tool that makes it easy to create intelligent and beautiful documentation, for execute the documentation you need to follow the next steps. U need a terminal to execute the commands in this directory.

1. Install the requirements
```bash
pip install -r requirements.txt
```

2. Build the documentation
```bash
make html
```

3. Open the documentation
```bash
open _build/html/index.html
```

4. Clean the documentation
```bash
make clean
```