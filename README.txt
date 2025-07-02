Master's project:
Title: Design and Implementation of a Photodiode Array-Based Analogue 2D Sun Sensor
Authors: Zac McCaffery, Alexandru Belea,Sebastian Alexander, William Kong, Nassor Salim

Download main.pdf for the full report.

Abstract

This research project explores the design, implementation, and testing of a prototype of a
cost-effective photodiode array-based analogue 2D sun sensor for attitude determination
in Low Earth Orbit (LEO) nanosatellites, with a focus on the CubeSat variety. As the
commercialisation of space continues to grow, there is a demand for low-cost, reliable at-
titude determination systems for small satellites that cannot accommodate the expensive
digital camera systems used in larger commercial missions. A comprehensive review of
existing sun sensor technologies, CubeSat designs, mechanical considerations and signal
analysis methods has been undertaken.
Based on this foundation, a prototype was developed utilising four photodiodes arranged
in a T-shaped configuration with appropriate apertures, coupled with transimpedance and
secondary amplification circuitry to process the photodiode signals. The prototype was
housed in a custom-designed 3D-printed enclosure to ensure proper positioning of the
photodiodes and protection of the electronic components. A software model was created
in parallel to simulate ray projection and intersection calculations, allowing for the pre-
diction of sensor response under various light conditions. This model provided valuable
insights for optimising the physical design and interpreting experimental results. Addi-
tionally, material analysis was performed to evaluate suitable materials for space deploy-
ment. Polyimide was identified as the optimal PCB material due to its balance of thermal
stability, radiation resistance, and mechanical properties. Thermal analysis using ANSYS
confirmed that the selected components would function reliably within the extreme tem-
perature range of space environments from 200C to -200C. A Data Acquisition System
(DAQ) based on an Arduino microcontroller was implemented to record and process the
sensor data, incorporating digital filtering to eliminate noise and enhance signal quality.
Testing was conducted using a Renewable Energy Demonstrator (RED) as a testbench
to position a light source at precise angles. The research demonstrates the feasibility
of developing a low-cost sun-sensing solution for nanosatellites that balances simplicity
with adequate performance for attitude determination in space applications. The findings
contribute to the growing field of small satellite technology and offer potential pathways
for future improvements in sun sensor design for space missions.

Compilation instructions:

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

I recommend VS Code as the editor with LaTeX Workshop extension for auto fill.


OR To compile in Overleaf:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
or zip the following and upload to overleaf:

bibliography\
chapters\
figures\
main.tex

^ you only do this once,

then you just would have to drag and drop any files you modify
in the repo to overleaf and Ctrl+S (and keep the repo uptodate for the rest of us!)

command to see number of words:

texcount -inc main.tex (in terminal)

and look for line "Words in text: "
