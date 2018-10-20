"""
Directional Variogram
"""
import numpy as np
from shapely.geometry import Polygon

from .Variogram import Variogram


class DirectionalVariogram(Variogram):
    """DirectionalVariogram Class

    Calculates a variogram of the separating distances in the given
    coordinates and relates them to one of the semi-variance measures of the
    given dependent values.

    The direcitonal version of a Variogram will only form paris of points
    that share a specified spatial relationship.

    """
    def __init__(self,
                 coordinates=None,
                 values=None,
                 estimator='matheron',
                 model='spherical',
                 dist_func='euclidean',
                 bin_func='even',
                 normalize=True,
                 fit_method='trf',
                 fit_sigma=None,
                 directional_model='triangle',
                 azimuth=0,
                 tolerance=45.0,
                 bandwidth=None,
                 use_nugget=False,
                 maxlag=None,
                 n_lags=10,
                 verbose=False,
                 harmonize=False
                 ):
        r"""Variogram Class

        Note: The directional variogram estimation is not re-implemented yet.
        At current stage it is just a skeleton for implementing the functions
        in the next step.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Array of shape (m, n). Will be used as m observation points of
            n-dimensions. This variogram can be calculated on 1 - n
            dimensional coordinates. In case a 1-dimensional array is passed,
            a second array of same length containing only zeros will be
            stacked to the passed one.
        values : numpy.ndarray
            Array of values observed at the given coordinates. The length of
            the values array has to match the m dimension of the coordinates
            array. Will be used to calculate the dependent variable of the
            variogram.
        estimator : str, callable
            String identifying the semi-variance estimator to be used.
            Defaults to the Matheron estimator. Possible values are:

              * matheron        [Matheron, default]
              * cressie         [Cressie-Hawkins]
              * dowd            [Dowd-Estimator]
              * genton          [Genton]
              * minmax          [MinMax Scaler]
              * entropy         [Shannon Entropy]

            If a callable is passed, it has to accept an array of absoulte
            differences, aligned to the 1D distance matrix (flattened upper
            triangle) and return a scalar, that converges towards small
            values for similarity (high covariance).
        model : str
            String identifying the theoretical variogram function to be used
            to describe the experimental variogram. Can be one of:

              * spherical       [Spherical, default]
              * exponential     [Exponential]
              * gaussian        [Gaussian]
              * cubic           [Cubic]
              * stable          [Stable model]
              * matern          [Matérn model]
              * nugget          [nugget effect variogram]

        dist_func : str
            String identifying the distance function. Defaults to
            'euclidean'. Can be any metric accepted by
            scipy.spatial.distance.pdist. Additional parameters are not (yet)
            passed through to pdist. These are accepted by pdist for some of
            the metrics. In these cases the default values are used.
        bin_func : str
            String identifying the binning function used to find lag class
            edges. At the moment there are two possible values: 'even'
            (default) or 'uniform'. Even will find n_lags bins of same width
            in the interval [0,maxlag[. 'uniform' will identfy n_lags bins on
            the same interval, but with varying edges so that all bins count
            the same amount of observations.
        normalize : bool
            Defaults to False. If True, the independent and dependent
            variable will be normalized to the range [0,1].
        fit_method : str
            String identifying the method to be used for fitting the
            theoretical variogram function to the experimental. More info is
            given in the Variogram.fit docs. Can be one of:

                * 'lm': Levenberg-Marquardt algorithm for unconstrained
                  problems. This is the faster algorithm, yet is the fitting of
                  a variogram not unconstrianed.
                * 'trf': Trust Region Reflective function for non-linear
                  constrained problems. The class will set the boundaries
                  itself. This is the default function.

        fit_sigma : numpy.ndarray, str
            Defaults to None. The sigma is used as measure of uncertainty
            during variogram fit. If fit_sigma is an array, it has to hold
            n_lags elements, giving the uncertainty for all lags classes. If
            fit_sigma is None (default), it will give no weight to any lag.
            Higher values indicate higher uncertainty and will lower the
            influcence of the corresponding lag class for the fit.
            If fit_sigma is a string, a pre-defined function of separating
            distance will be used to fill the array. Can be one of:

                * 'linear': Linear loss with distance. Small bins will have
                  higher impact.
                * 'exp': The weights decrease by a e-function of distance
                * 'sqrt': The weights decrease by the squareroot of distance
                * 'sq': The weights decrease by the squared distance.

            More info is given in the Variogram.fit_sigma documentation.
        directional_model : string, Polygon
            The model used for selecting all points fulfilling the
            directional constraint of the Variogram. A predefined model can
            be selected by passing the model name as string. Optionally a
            callable accepting the current local coordinate system and
            returning a Polygon representing the search area itself
            can be passed. In this case, the tolerance and bandwidth has to
            be incorporated by hand into the model. The azimuth is handled
            by the class. The predefined options are:

                * 'compass': includes points in the direction of the
                  azimuth at given tolerance. The bandwidth parameter will be
                  ignored.
                * 'triangle': constructs a triangle with an angle of
                  tolerance at the point of interest and union an rectangle
                  parallel to azimuth, once the hypotenuse length reaches
                  bandwidth.
                * 'circle': constructs a half circle touching the point of
                  interest, dislocating the center at the distance of
                  bandwidth in the direction of azimuth. The half circle is
                  union with an rectangle parallel to azimuth.

            Visual representations, usage hints and implementation specifics
            are given in the documentation.
        azimuth : float
            The azimuth of the directional dependence for this Variogram,
            given as an angle in **degree**. The North of the coordinate
            plane is set to be at 0° and is counted clockwise to 360°
            (which is North again). Only Points lying in the azimuth of a
            specific point will be used for forming point pairs.
        tolerance : float
            The tolerance is given as an angle in **degree**- Points being
            dislocated from the exact azimuth by half the tolerance will be
            accepted as well. It's half the tolerance as the pointmay be
            dislocated in the positive and negative direction from the azimuth.
        bandwidth : float
            Maximum tolerance acceptable in **coordinate units**, which is
            usually meter. Points at higher distances may be far dislocated
            from the azimuth in terms of coordinate distance, as the
            tolerance is defined as an angle. THe bandwidth defines a maximum
            width for the search window. It will be perpendicular to and
            bisected by the azimuth.
        use_nugget : bool
            Defaults to False. If True, a nugget effet will be added to all
            Variogram.models as a third (or fourth) fitting parameter. A
            nugget is essentially the y-axis interception of the theoretical
            variogram function.
        maxlag : float, str
            Can specify the maximum lag distance directly by giving a value
            larger than 1. The binning function will not find any lag class
            with an edge larger than maxlag. If 0 < maxlag < 1, then maxlag
            is relative and maxlag * max(Variogram.distance) will be used.
            In case maxlag is a string it has to be one of 'median', 'mean'.
            Then the median or mean of all Variogram.distance will be used.
            Note maxlag=0.5 will use half the maximum separating distance,
            this is not the same as 'median', which is the median of all
            separating distances
        n_lags : int
            Specify the number of lag classes to be defined by the binning
            function.
        verbose : bool
            Set the Verbosity of the class. Not Implemented yet.
        harmonize : bool
            this kind of works so far, but will be rewritten (and documented)
        """
        # Set coordinates
        self._X = np.asarray(coordinates)

        # pairwise difference
        self._diff = None

        # set verbosity
        self.verbose = verbose

        # set values
        self._values = None
        self.set_values(values=values)

        # distance matrix
        self._dist = None

        # set distance calculation function
        self._dist_func = None
        self.set_dist_function(func=dist_func)

        # lags and max lag
        self.n_lags = n_lags
        self._maxlag = None
        self.maxlag = maxlag

        # estimator can be function or a string
        self._estimator = None
        self.set_estimator(estimator_name=estimator)

        # model can be function or a string
        self._model = None
        self.set_model(model_name=model)

        # azimuth direction
        self._azimuth = None
        self.azimuth = azimuth

        # azimuth tolerance
        self._tolerance = None
        self.tolerance = tolerance

        # tolerance bandwidth
        self._bandwidth = None
        self.bandwidth = bandwidth

        # set the directional model
        self._directional_model = None
        self.set_directional_model(model_name=directional_model)

        # the binning settings
        self._bin_func = None
        self._groups = None
        self._bins = None
        self.set_bin_func(bin_func=bin_func)

        # specify if the lag should be given absolute or relative to the maxlag
        self._normalized = normalize

        # specify if the experimental variogram shall be harmonized
        self.harmonize = harmonize

        # set the fitting method and sigma array
        self.fit_method = fit_method
        self._fit_sigma = None
        self.fit_sigma = fit_sigma

        # set if nugget effect shall be used
        self.use_nugget = use_nugget

        # set attributes to be filled during calculation
        self.cov = None
        self.cof = None

        # settings, not reachable by init (not yet)
        self._cache_experimental = False

        # do the preprocessing and fitting upon initialization
        self.preprocessing(force=True)
        self.fit(force=True)

    @property
    def azimuth(self):
        return self._azimuth

    @azimuth.setter
    def azimuth(self, angle):
        """Direction azimuth

        Main direction for te selection of points in the formation of point
        pairs. North of the coordinate plane is defined to be 0° and then the
        azimuth is set clockwise up to 360°, which is North again.

        Parameters
        ----------
        angle : float
            New azimuth angle in **degree**.

        Raises
        ------
        ValueError : in case angle < 0 or angle > 360

        """
        if angle < 0 or angle > 360:
            raise ValueError('The azimuth is an angle in degree and has to '
                             'meet 0 <= angle <= 360')
        else:
            self._azimuth = angle

    @property
    def tolerance(self):
        return self._tolerance

    @tolerance.setter
    def tolerance(self, angle):
        """Azimuth tolerance

        Tolerance angle of how far a point can be off the azimuth for being
        still counted as directional. A tolerance angle will be applied to
        the azimuth angle symmetrically.

        Parameters
        ----------
        angle : float
            New tolerance angle in **degree**. Has to meet 0 <= angle <= 360.

        Raises
        ------
        ValueError : in case angle < 0 or angle > 360

        """
        if angle < 0 or angle > 360:
            raise ValueError('The tolerance is an angle in degree and has to '
                             'meet 0 <= angle <= 360')
        else:
            self._tolerance = angle

    @property
    def bandwidth(self):
        if self._bandwidth is None:
            return 0
        else:
            return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, width):
        """Tolerance bandwidth

        New bandwidth parameter. As the tolerance from azimuth is given as an
        angle, point pairs at high distances can be far off the azimuth in
        coordinate distance. The bandwidth limits this distance and has the
        unnit of the coordinate system.

        Parameters
        ----------
        width : float
            Positive coordinate distance.

        Raises
        ------
        ValueError : in case width is negative

        """
        if width < 0:
            raise ValueError('The bandwidth cannot be negative.')
        elif width > np.max(self.distance):
            print('The bandwidth is larger than the maximum separating '
                  'distance. Thus it will have no effect.')
        else:
            self._bandwidth = width

    def set_directional_model(self, model_name):
        """Set new directional model

        The model used for selecting all points fulfilling the
        directional constraint of the Variogram. A predefined model can
        be selected by passing the model name as string. Optionally a function
        can be passed that accepts the current local coordinate system and
        returns a Polygon representing the search area. In this case, the
        tolerance and bandwidth has to be incorporated by hand into the
        model. The azimuth is handled by the class. The predefined options are:

        * 'compass': includes points in the direction of the azimuth at given
           tolerance. The bandwidth parameter will be ignored.
        * 'triangle': constructs a triangle with an angle of tolerance at the
          point of interest and union an rectangle parallel to azimuth,
          once the hypotenuse length reaches bandwidth.
        * 'circle': constructs a half circle touching the point of interest,
          dislocating the center at the distance of bandwidth in the
          direction of azimuth. The half circle is union with an rectangle
          parallel to azimuth.

        Visual representations, usage hints and implementation specifics
        are given in the documentation.

        Parameters
        ----------
        model_name : string, callable
            The name of the predefined model (string) or a function that
            accepts the current local coordinate system and returns a Polygon
            of the search area.

        """
        # handle predefined models
        if isinstance(model_name, str):
            if model_name.lower() == 'compass':
                raise NotImplementedError
            elif model_name.lower() == 'triangle':
                pass
            elif model_name.lower() == 'circle':
                raise NotImplementedError
            else:
                raise ValueError('%s is not a valid model.' % model_name)

        # handle Polygons
        elif isinstance(model_name, Polygon):
            self._directional_model = model_name
        else:
            raise ValueError('The directional model has to be identified by a '
                             'model name, or it has to be the search area '
                             'itself')

    def _triangle(self, local_ref):
        r"""Triangular Search Area

        Construct a triangular bounded search area for building directional
        dependent point pairs. The Search Area will be located onto the
        current point of interest and the local x-axis is rotated onto the
        azimuth angle.

        Parameters
        ----------
        local_ref : numpy.array
            Array of all coordinates transformed into a local representation
            with the current point of interest being the origin and the
            azimuth angle aligned onto the x-axis.

        Returns
        -------
        search_area : Polygon
            Search Area of triangular shape bounded by the current bandwidth.

        Notes
        -----
               C
              /|\
           a / | \ a
            /__|h_\
           A   c   B

        The point of interest is C and c is the bandwidth. The angle at C
        (gamma) is the tolerance. From this, a and then h can be calculated.
        When rotated into the local coordinate system, the two points needed
        to build the search area A,B are A := (h, 1/2 c) and B:= (h, -1/2 c)

        a can be calculated like:

        .. math::
            a = \frac{c}{2 * sin\left(\frac{\gamma}{2}\right)}

        """
        # y coordinate is easy
        c = self.bandwidth
        gamma = np.radians(self.tolerance)
        y = 0.5 * c

        # a and h can be calculated
        a = c / (2 * np.sin(gamma / 2))
        h = np.sqrt((4 * a**2 - c**2) / 4)

        # get the maximum x coordinate in the current representation
        xmax = np.max(local_ref[:, 0])

        # build the figure
        poly = Polygon([
            (0, 0),
            (h, y),
            (xmax, y),
            (xmax, -y),
            (h, -y)
        ])

        return poly
