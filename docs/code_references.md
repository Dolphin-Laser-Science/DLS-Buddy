# Code References

Every external source cited anywhere in DLS Buddy (source code or the
Advanced Guide), mapped to **where it is used** — `path:line` for code, a
`§section`/equation for the guide. Use it to trace any physical claim back
to its literature, or to find which papers a given module relies on.

> Generated from the project bibliography; do not edit by hand.

---

## Literature and manuals

### Dynamic light scattering

- Koppel, Dennis E. (1972). Analysis of Macromolecular Polydispersity in Intensity Correlation Spectroscopy: The Method of Cumulants. *Journal of Chemical Physics* **57**(11), 4814–4820. https://doi.org/10.1063/1.1678153
  - Used in: analysis/dls/cumulants.py; analysis/dls/__init__.py (module docstring)
- Frisken, Barbara J. (2001). Revisiting the Method of Cumulants for the Analysis of Dynamic Light-Scattering Data. *Applied Optics* **40**(24), 4087–4091. https://doi.org/10.1364/AO.40.004087
  - Used in: analysis/dls/cumulants.py (_fit_cumulants_nonlinear, _frisken_model); Advanced Guide §10.1, Eq. (47)
- Provencher, Stephen W. (1982). A Constrained Regularization Method for Inverting Data Represented by Linear Algebraic or Integral Equations. *Computer Physics Communications* **27**(3), 213–227. https://doi.org/10.1016/0010-4655(82)90173-4
  - Used in: analysis/dls/distributions.py (_ftest_corner, _tikhonov_effective_dof – the F-test alpha-selection criterion Eqs. 3.23–3.24 and the effective degrees of freedom NDF Eqs. 3.15–3.16); Advanced Guide §10.2, Eqs. (48)–(49)
- Provencher, Stephen W. (1982). CONTIN: A General Purpose Constrained Regularization Program for Inverting Noisy Linear Algebraic and Integral Equations. *Computer Physics Communications* **27**(3), 229–242. https://doi.org/10.1016/0010-4655(82)90174-6
  - Used in: analysis/dls/distributions.py (fit_contin – CONTIN Tikhonov inversion); analysis/dls/__init__.py (module docstring); Advanced Guide §10.2, Eq. (17)
- Scotti, A.; Liu, W.; Hyatt, J. S.; Herman, E. S.; Choi, H. S.; Kim, J. W.; Lyon, L. A.; Gasser, U.; Fernandez-Nieves, A. (2015). The CONTIN Algorithm and Its Application to Determine the Size Distribution of Microgel Suspensions. *The Journal of Chemical Physics* **142**(23), 234905. https://doi.org/10.1063/1.4921686
  - Used in: analysis/dls/distributions.py (_ftest_corner, _tikhonov_effective_dof – probability-to-reject F-test, Eqs. 19–21); Advanced Guide §10.2, Eqs. (48)–(49)
- Hansen, Per Christian (1998). Rank-Deficient and Discrete Ill-Posed Problems: Numerical Aspects of Linear Inversion. SIAM. https://doi.org/10.1137/1.9780898719697
  - Used in: analysis/dls/distributions.py (_tikhonov_effective_dof – hat-matrix trace / effective degrees of freedom); Advanced Guide §10.2, Eq. (49)
- Salazar, Marcos; Srivastav, Harsh; Srivastava, Abhishek; Srivastava, Samanvaya (2023). A User-Friendly Graphical User Interface for Dynamic Light Scattering Data Analysis. *Soft Matter* **19**(35), 6535–6544. https://doi.org/10.1039/d3sm00469d
  - Used in: analysis/dls/distributions.py (_lcurve_corner); analysis/dls/__init__.py (module docstring)
- Liénard, François; Freyssingeas, Éric; Borgnat, Pierre (2022). A Multiscale Time-Laplace Method to Extract Relaxation Times from Non-Stationary Dynamic Light Scattering Signals. *Journal of Chemical Physics* **156**(22), 224901. https://doi.org/10.1063/5.0088005
  - Used in: analysis/dls/distributions.py:13 (cross-terms-negligible approximation, Eq. 11)

### Static light scattering

- Takahashi, Kazuki; Takano, Atsushi; Kinugasa, Shinichi; Sakurai, Hiroshi (2019). Determination of the Rayleigh Ratio of Toluene with an Uncertainty Analysis. *Analytical Sciences* **35**(9), 1045–1051. https://doi.org/10.2116/analsci.19P103
  - Used in: physics/constants.py:263–265; physics/constants.py:272; physics/constants.py:306; analysis/sls.py:19–20; analysis/sls.py:65
- Sivokhin, Alexey A.; Kazantsev, Oleg A. (2021). Temperature Dependence of the Rayleigh Ratio of Toluene and Its Depolarization Ratio. *ChemistrySelect* **6**(35), 9499–9502. https://doi.org/10.1002/slct.202102196
  - Used in: physics/constants.py:266–268; physics/constants.py:277; physics/constants.py:284; physics/constants.py:306; physics/constants.py:330; physics/constants.py:565; physics/constants.py:606; analysis/sls.py:51; analysis/sls.py:67; core/data_models.py:540–541; parsers/brookhaven_sls.py:60; parsers/brookhaven_sls.py:314; Advanced Guide §11.6, Eq. (38)
- Wu, Hua (2010). Correlations between the Rayleigh Ratio and the Wavelength for Toluene and Benzene. *Chemical Physics* **367**(1), 44–47. https://doi.org/10.1016/j.chemphys.2009.10.019
  - Used in: physics/constants.py:258; physics/constants.py:270; physics/constants.py:279
- Seery, Thomas A. P.; Shorter, John A.; Amis, Eric J. (1989). Concurrent Static and Dynamic Light Scattering from Macromolecular Solutions. 1. Model Systems in the Low q Regime. *Polymer* **30**(7), 1197–1203. https://doi.org/10.1016/0032-3861(89)90036-0
  - Used in: analysis/sls.py:43; analysis/sls.py:66
- Guinier, André (1939). La diffraction des rayons X aux très petits angles: application à l'étude de phénomènes ultramicroscopiques. *Annales de Physique (Paris)* **11**(12), 161–237. https://doi.org/10.1051/anphys/193911120161
  - Used in: analysis/sls.py (guinier_analysis); Advanced Guide §11.5, Eq. (27)
- Russo, Paul S.; Streletzky, Kiril A.; Huberty, Wayne; Zhang, Xujun; Edwin, Nadia (2021). Characterization of Polymers by Static Light Scattering. Ch. 13 in Molecular Characterization of Polymers: A Fundamental Guide, eds. Malik, Mays & Shah (Elsevier). ISBN 9780128197684; accessible derivation of the Guinier plot (static light scattering chapter).
  - Used in: analysis/sls.py (guinier_analysis); Advanced Guide §11.5 (Guinier plot)

### Depolarized light scattering (static)

- Chu, Benjamin (1991). Laser Light Scattering: Basic Principles and Practice. 2 ed. Academic Press. Section 8.4.1.A, pp. 290–291, Eqs. 8.4.7–8.4.10.
  - Used in: physics/constants.py:639; physics/constants.py:730; physics/constants.py (scattering_vector_q, stokes_einstein_rh; the scattering vector q and Stokes-Einstein relation); analysis/depolarization.py; analysis/utilities.py (_interpret_rho, rho_shape_label; rho=Rg/Rh reference values); analysis/sls.py (guinier_analysis; the Guinier plot); analysis/dls/exponentials.py + analysis/dls/distributions.py (the Siegert relation; historical origin A. J. F. Siegert, Report No. 465, Radiation Laboratory, MIT, 1943); Advanced Guide §7.1 (Eq. 3), §7.2 (Eq. 4), §9.4 (Eq. 10), §10 (Eq. 2), §11.5, and §11.6 (Eqs. 39–40)
- Coumou, D. J.; Mackor, E. L.; Hijmans, J. (1964). Isotropic Light Scattering in Pure Liquids. *Transactions of the Faraday Society* **60**, 1539–1547. https://doi.org/10.1039/TF9646001539
  - Used in: physics/constants.py:689; Advanced Guide §11.6, Eq. (41)

### Depolarized light scattering (dynamic / DDLS)

- Pecora, Robert (1964). Doppler Shifts in Light Scattering from Pure Liquids and Polymer Solutions. *Journal of Chemical Physics* **40**(6), 1604–1614. https://doi.org/10.1063/1.1725368
  - Used in: analysis/depolarization.py (analyze_ddls, rotational_diffusion_from_rates); app/controller.py (run_ddls); Advanced Guide §10.4, Eqs. (42)–(43)
- Zero, Kenneth; Pecora, Robert (1982). Rotational and Translational Diffusion in Semidilute Solutions of Rigid-Rod Macromolecules. *Macromolecules* **15**(1), 87–93. https://doi.org/10.1021/ma00229a018
  - Used in: analysis/depolarization.py (analyze_ddls, the qL guard); Advanced Guide §10.4, Eq. (42)
- Tirado, María M.; López Martínez, Carmen; García de la Torre, José (1984). Comparison of Theories for the Translational and Rotational Diffusion Coefficients of Rod-Like Macromolecules. Application to Short DNA Fragments. *Journal of Chemical Physics* **81**(4), 2047–2052. https://doi.org/10.1063/1.447827
  - Used in: physics/constants.py (rod_translational_diffusion, rod_rotational_diffusion, rod_end_corrections, rod_length_from_translational_diffusion); analysis/depolarization.py (rod_dimensions_from_diffusion); Advanced Guide §10.5, Eqs. (45)–(46)
- Balog, Sandor; Rodriguez-Lorenzo, Laura; Monnier, Christophe A.; Obiols-Rabasa, Marc; Rothen-Rutishauser, Barbara; Schurtenberger, Peter; Petri-Fink, Alke (2015). Characterizing Nanoparticles in Complex Biological Media and Physiological Fluids with Depolarized Dynamic Light Scattering. *Nanoscale* **7**(14), 5991–5997. https://doi.org/10.1039/c4nr06538g
  - Used in: physics/constants.py (sphere_rotational_diffusion, sphere_radius_from_rotational_diffusion); analysis/depolarization.py (sphere_dimensions_from_diffusion); Advanced Guide §10.5, Eq. (44)

### Polymer physics (scaling)

- Rubinstein, Michael; Colby, Ralph H. (2003). Polymer Physics. Oxford University Press.
  - Used in: analysis/utilities.py (fit_power_law, ScalingResult, interpret_scaling_exponent); Advanced Guide §9.5, Eqs. (28)–(29)
- Fetters, Lewis J.; Hadjichristidis, Nikos; Lindner, Jimmy S.; Mays, Jimmy W. (1994). Molecular Weight Dependence of Hydrodynamic and Thermodynamic Properties for Well-Defined Linear Polymers in Solution. *Journal of Physical and Chemical Reference Data* **23**(4), 619–640. https://doi.org/10.1063/1.555949
  - Used in: analysis/utilities.py (interpret_scaling_exponent); Advanced Guide §9.5

### Uncertainty and statistics

- MacKinnon, James G.; White, Halbert (1985). Some Heteroskedasticity-Consistent Covariance Matrix Estimators with Improved Finite Sample Properties. *Journal of Econometrics* **29**(3), 305–325. https://doi.org/10.1016/0304-4076(85)90158-7
  - Used in: analysis/uncertainty.py (_hc3_cov, module docstring); Advanced Guide §15.1, Eq. (30)
- Long, J. Scott; Ervin, Laurie H. (2000). Using Heteroscedasticity Consistent Standard Errors in the Linear Regression Model. *The American Statistician* **54**(3), 217–224. https://doi.org/10.1080/00031305.2000.10474549
  - Used in: analysis/uncertainty.py (module docstring); Advanced Guide §15.1
- Draper, Norman R.; Smith, Harry (1998). Applied Regression Analysis. 3 ed. Wiley. https://doi.org/10.1002/9781118625590
  - Used in: analysis/uncertainty.py (_ols_cov); Advanced Guide §15.1.1, Eq. (30a)
- Taylor, John R. (1997). An Introduction to Error Analysis: The Study of Uncertainties in Physical Measurements. 2 ed. University Science Books.
  - Used in: analysis/uncertainty.py (propagate, ratio_se, power_law_se); Advanced Guide §15.2, Eqs. (33)–(35)
- Bevington, Philip R.; Robinson, D. Keith (2003). Data Reduction and Error Analysis for the Physical Sciences. 3 ed. McGraw-Hill.
  - Used in: Advanced Guide §15.2
- Schätzel, Klaus (1990). Noise on Photon Correlation Data: I. Autocorrelation Functions. *Quantum Optics* **2**(4), 287–305. https://doi.org/10.1088/0954-8998/2/4/002
  - Used in: analysis/uncertainty.py (module docstring, replicate_mean_se); analysis/dls/replicate.py (average_replicate_correlograms); Advanced Guide §15.4
- International Organization for Standardization (2017). Particle Size Analysis – Dynamic Light Scattering (DLS). ISO 22412:2017, Geneva.
  - Used in: analysis/uncertainty.py:replicate_mean_se (Eq. 37); app/controller.py:average_dls_results; Advanced Guide §15.4, Eq. (37)

### Fundamental constants

- CODATA / NIST (2024). CODATA Recommended Values of the Fundamental Physical Constants: 2022. NIST SP 961 (May 2024), National Institute of Standards and Technology. k_B and N_A are exact values fixed by the 2019 SI redefinition.
  - Used in: physics/constants.py:70; physics/constants.py:73 (k_B = 1.380649e-23 J/K; N_A = 6.02214076e23 /mol)

### Instrument manuals

- Brookhaven BI-200SM Manual, Section VIII. Brookhaven Instruments goniometer/correlator manual.
  - Used in: physics/constants.py:258; analysis/sls.py:63 (R_VU = R_VV(1+rho_v) geometry; Rayleigh calibration)
- Brookhaven BIZPW (Zimm Plot) Manual, Eq. 3. Brookhaven Instruments software manual.
  - Used in: analysis/sls.py:19; analysis/sls.py:64 (basic Zimm SLS equation)
- Brookhaven Particle Explorer Manuals. Brookhaven Instruments software manuals (DLS correlogram / SLS intensity export formats).
  - Used in: analysis/dls/__init__.py (module docstring); analysis/sls.py:64; parsers/brookhaven_dls.py; parsers/brookhaven_sls.py
- Brookhaven TurboCorr DLSW Manual. Brookhaven Instruments correlator manual (DLS conventions cross-reference).
  - Used in: analysis/dls/__init__.py (module docstring)

## Named methods and relations

Standard, named techniques used in the code. Not always tied to a single paper
in-source; listed so a forker can find where each lives.

| Method / relation | Where used | Note |
|---|---|---|
| **Stokes–Einstein** relation | `physics/constants.py:144–239` | `Rh = k_B T / (6πηD)` (Chu 1991) |
| **Siegert** relation | `analysis/dls/` (cumulants, exponentials, distributions); `analysis/utilities.py` (`generate_synthetic_correlogram`); `parsers/generic_dls.py:36,176` | `g₂−1 = β\|g₁\|²` (Chu 1991; orig. Siegert 1943) |
| **Zimm / Berry / Debye** plots | `analysis/sls.py` (Zimm/Berry `:497–664`, Debye `:404–445`) | Berry = √-ordinate variant for high `qRg` |
| **Guinier** plot | `analysis/sls.py` (`guinier_analysis`) | ln(ΔR) vs q²; Rg from slope (Guinier 1939) |
| **Kohlrausch–Williams–Watts (KWW)** | `analysis/dls/exponentials.py` (`fit_kww`) | stretched exponential |
| **Tikhonov regularisation** | `analysis/dls/distributions.py` (CONTIN, `fit_contin`) | 2nd-difference operator; L-curve corner |
| **Non-negative least squares (NNLS)** | `analysis/dls/distributions.py` (`fit_nnls`) | `scipy.optimize.nnls` |
| **Freedman–Diaconis** rule | `analysis/trace_analysis.py` (`_freedman_diaconis_bins`) | histogram bin width |
| **Sturges'** rule | `analysis/trace_analysis.py` (`_sturges_bins`) | histogram bin fallback |
| **Augmented Dickey–Fuller (ADF)** test | `analysis/trace_analysis.py` (`test_stationarity_adf`) | stationarity (`statsmodels`) |
| **HC3 heteroscedasticity-consistent SE** | `analysis/uncertainty.py` (`_hc3_cov`) | robust regression covariance (MacKinnon & White 1985; Long & Ervin 2000) |
| **Delta method (1st-order propagation)** | `analysis/uncertainty.py` (`propagate`, `ratio_se`, `power_law_se`) | `Var(f)=JᵀΣJ` (Taylor 1997; Bevington & Robinson 2003) |
