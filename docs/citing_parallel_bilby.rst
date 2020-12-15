=======================================
Acknowledging/Citing Parallel Bilby
=======================================

If you have used Parallel Bilby in your scientific work we would appreciate it if you would acknowledge it. The continued growth and development of Parallel Bilby is dependent on the community being aware of Parallel Bilby.

Publications
------------
Please add the following line within your methods, conclusion or acknowledgements
sections:

    "This research has made use of Parallel Bilby vX.Y \cite{pbilby_paper, bilby_paper},
    a parallelised Bayesian inference Python package, and Dynesty vX.Y
    \cite{dynesty_paper,  skilling2004, skilling2006}, a nested sampler,
    to perform Bayesian parameter estimation."

.. code:: bibtex

    @ARTICLE{pbilby_paper,
           author = {{Smith}, Rory J.~E. and {Ashton}, Gregory and {Vajpeyi}, Avi and {Talbot}, Colm},
            title = "{Massively parallel Bayesian inference for transient gravitational-wave astronomy}",
          journal = {\mnras},
         keywords = {gravitational waves, methods: data analysis, General Relativity and Quantum Cosmology, Astrophysics - Instrumentation and Methods for Astrophysics},
             year = 2020,
            month = aug,
           volume = {498},
           number = {3},
            pages = {4492-4502},
              doi = {10.1093/mnras/staa2483},
    archivePrefix = {arXiv},
           eprint = {1909.11873},
     primaryClass = {gr-qc},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.498.4492S},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{bilby_paper,
           author = {{Ashton}, Gregory and {H{\"u}bner}, Moritz and {Lasky}, Paul D. and {Talbot}, Colm and {Ackley}, Kendall and {Biscoveanu}, Sylvia and {Chu}, Qi and {Divakarla}, Atul and {Easter}, Paul J. and {Goncharov}, Boris and {Hernandez Vivanco}, Francisco and {Harms}, Jan and {Lower}, Marcus E. and {Meadors}, Grant D. and {Melchor}, Denyz and {Payne}, Ethan and {Pitkin}, Matthew D. and {Powell}, Jade and {Sarin}, Nikhil and {Smith}, Rory J.~E. and {Thrane}, Eric},
            title = "{BILBY: A User-friendly Bayesian Inference Library for Gravitational-wave Astronomy}",
          journal = {\apjs},
         keywords = {gravitational waves, methods: data analysis, methods: statistical, stars: black holes, stars: neutron, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - High Energy Astrophysical Phenomena, General Relativity and Quantum Cosmology},
             year = 2019,
            month = apr,
           volume = {241},
           number = {2},
              eid = {27},
            pages = {27},
              doi = {10.3847/1538-4365/ab06fc},
    archivePrefix = {arXiv},
           eprint = {1811.02042},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2019ApJS..241...27A},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{dynesty_paper,
           author = {{Speagle}, Joshua S.},
            title = "{DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences}",
          journal = {\mnras},
         keywords = {methods: data analysis, methods: statistical, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Computation},
             year = 2020,
            month = apr,
           volume = {493},
           number = {3},
            pages = {3132-3158},
              doi = {10.1093/mnras/staa278},
    archivePrefix = {arXiv},
           eprint = {1904.02180},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @INPROCEEDINGS{skilling2004,
           author = {{Skilling}, John},
            title = "{Nested Sampling}",
         keywords = {02.50.Tt, Inference methods},
        booktitle = {Bayesian Inference and Maximum Entropy Methods in Science and Engineering: 24th International Workshop on Bayesian Inference and Maximum Entropy Methods in Science and Engineering},
             year = 2004,
           editor = {{Fischer}, Rainer and {Preuss}, Roland and {Toussaint}, Udo Von},
           series = {American Institute of Physics Conference Series},
           volume = {735},
            month = nov,
            pages = {395-405},
              doi = {10.1063/1.1835238},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2004AIPC..735..395S},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }


    @ARTICLE{skilling2006,
           author = {{Skilling}, John},
              doi = {10.1214/06-BA127},
          journal = {Bayesian Analysis},
          journal = {Bayesian Analysis},
            month = dec,
           number = 4,
            pages = {833-859},
        publisher = {International Society for Bayesian Analysis},
            title = {Nested sampling for general Bayesian computation},
              url = "https://doi.org/10.1214/06-BA127",
           volume = 1,
             year = 2006
    }



The citation should be to the `Parallel Bilby paper`_ and the version number should
cite the specific version used in your work.


Posters and talks
------------------
Please include the `Parallel Bilby logo`_ on the title, conclusion slide, or about page.


.. _Parallel Bilby paper: http://dx.doi.org/10.1093/mnras/staa2483
.. _Parallel Bilby logo: https://git.ligo.org/uploads/-/system/project/avatar/1846/bilby.jpg?width=40
