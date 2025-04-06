
Install TeXworks full package:
Follow easy install from : https://tug.org/texlive/windows.html

To compile using TeXworks:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The complete sequence would be:
Select these next to the play button in order and hit play button:

pdfLaTeX
BibTeX (if using references)
pdfLaTeX
pdfLaTeX

This is needed because:

The first run generates auxiliary files
BibTeX processes citations
Second run uses the TOC/LOF information
Third run resolves all references

OR To compile in Overleaf:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
or zip the following and upload to overleaf:

bibliography\
chapters\
figures\
main.tex