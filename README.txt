to compile using TeXworks:
The complete sequence would be:

pdfLaTeX
BibTeX (if using references)
pdfLaTeX
pdfLaTeX

This is needed because:

The first run generates auxiliary files
BibTeX processes citations
Second run uses the TOC/LOF information
Third run resolves all references

