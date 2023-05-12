"""
.. module:: spectrum
   :synopsis: Defines the Spectrum class to store spectroscopic data.
"""
class Spectrum:
    """
    This class defines a Spectrum object to store spectroscopic data and metadata necessary to
    perform cross-correlation and derive radial velocities.
    """
    def __init__(self):
        """
        Class constructor.
        """
        # A spectrum is allowed to be in one or more "parts", loosely defined.  For example, each
        # detector in APOGEE can be considered a semi-independent "part" containing sets of
        # wavelengths and fluxes that can be cross-correlated indepdently, or individual orders of
        # an Echelle spectrum if the orders have not been stitched into a single spectrum.
        self.parts = []
        self.name = ""
        self.air_or_vac = ""
        self.obj_ra = -999.999
        self.obj_dec = -999.999
        self.juldate = -999.999
        self.bjuldate = -999.999

    def add_spec_part(self, wls, fls, flerrs=None):
        """
        Adds a new "part" of wavelengths, fluxes, and (optional) flux uncertainties.

        :param wls: The set of wavelengths, in Angstroms, for this part.
        :type wls: numpy.ndarray

        :param fls: The set of normalized fluxes for this part.
        :type fls: numpy.ndarray

        :param flerrs: [Optional] The set of flux errors for this part.
        :type flerrs: numpy.ndarray
        """

        self.parts.append({"wls":wls, "fls":fls, "flerrs":flerrs})
