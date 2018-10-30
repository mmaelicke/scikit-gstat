"""
Directional Variogram
"""
import numpy as np
from numba import jit
from shapely.geometry import Polygon, Point
from itertools import chain
import matplotlib.pyplot as plt

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
                 bandwidth='q33',
                 use_nugget=False,
                 maxlag=None,
                 n_lags=10,
                 verbose=False,
                 harmonize=False
                 ):
        r"""Variogram Class

        Directional Variogram. The calculation is not performant and not
        tested yet.

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
            given as an angle in **degree**. The East of the coordinate
            plane is set to be at 0° and is counted clockwise to 180° and
            counter-clockwise to -180°. Only Points lying in the azimuth of a
            specific point will be used for forming point pairs.
        tolerance : float
            The tolerance is given as an angle in **degree**- Points being
            dislocated from the exact azimuth by half the tolerance will be
            accepted as well. It's half the tolerance as the point may be
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
        """Direction azimuth

        Main direction for te selection of points in the formation of point
        pairs. East of the coordinate plane is defined to be 0° and then the
        azimuth is set clockwise up to 180°and count-clockwise to -180°.

        Parameters
        ----------
        angle : float
            New azimuth angle in **degree**.

        Raises
        ------
        ValueError : in case angle < -180° or angle > 180

        """
        return self._azimuth

    @azimuth.setter
    def azimuth(self, angle):
        if angle < -180 or angle > 180:
            raise ValueError('The azimuth is an angle in degree and has to '
                             'meet -180 <= angle <= 180')
        else:
            self._azimuth = angle

        # reset groups on azimuth change
        self._groups = None

    @property
    def tolerance(self):
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
        return self._tolerance

    @tolerance.setter
    def tolerance(self, angle):
        if angle < 0 or angle > 360:
            raise ValueError('The tolerance is an angle in degree and has to '
                             'meet 0 <= angle <= 360')
        else:
            self._tolerance = angle

        # reset groups on tolerance change
        self._groups = None

    @property
    def bandwidth(self):
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
        if self._bandwidth is None:
            return 0
        else:
            return self._bandwidth

    @bandwidth.setter
    def bandwidth(self, width):
        # check if quantiles is given
        if isinstance(width, str):
            # TODO document and handle more exceptions
            q = int(width[1:])
            self._bandwidth = np.percentile(self.distance, q)
        elif width < 0:
            raise ValueError('The bandwidth cannot be negative.')
        elif width > np.max(self.distance):
            print('The bandwidth is larger than the maximum separating '
                  'distance. Thus it will have no effect.')
        else:
            self._bandwidth = width

        # reset groups on bandwidth change
        self._groups = None

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
                self._directional_model = self._compass
            elif model_name.lower() == 'triangle':
                self._directional_model = self._triangle
            elif model_name.lower() == 'circle':
                self._directional_model = self._circle
            else:
                raise ValueError('%s is not a valid model.' % model_name)

        # handle callable
        elif callable(model_name):
            self._directional_model = model_name
        else:
            raise ValueError('The directional model has to be identified by a '
                             'model name, or it has to be the search area '
                             'itself')

        # reset the groups as the directional model changed
        self._groups = None

#    @jit
    def local_reference_system(self, poi):
        """Calculate local coordinate system

        The coordinates will be transformed into a local reference system
        that will simplify the directional dependence selection. The point of
        interest (poi) of the current iteration will be used as origin of the
        local reference system and the x-axis will be rotated onto the azimuth.

        Parameters
        ----------
        poi : tuple
            First two coordinate dimensions of the point of interest. will be
            used as the new origin

        Returns
        -------
        local_ref : numpy.array
            Array of dimension (m, 2) where m is the length of the
            coordinates array. Transformed coordinates in the same order as
            the original coordinates.

        """
        # define a point-wise transform function
        def _transform(p1, p2, a):
            p = p1 - p2
            x = p[0] * np.cos(a) - p[1] * np.sin(a)
            y = p[0] * np.sin(a) + p[1] * np.cos(a)
            return np.array([x, y])

        # get the azimuth in radians
        gamma = np.radians(self.azimuth)

        # transform
        _X = np.fromiter(chain.from_iterable(
            map(lambda p: _transform(p, poi, gamma), self._X)
        ), dtype=self._X.dtype).reshape(self._X.shape)

        # return
        return _X

    @property
    def bins(self):
        if self._bins is None:
            # get the distances
            d = self.distance.copy()
            d[np.where(~self._direction_mask())] = np.nan

            self._bins = self.bin_func(d, self.n_lags, self.maxlag)

        return self._bins.copy()

    def _calc_groups(self, force=False):
        super(DirectionalVariogram, self)._calc_groups(force=force)

        # set to outside maxlag group
        self._groups[np.where(~self._direction_mask())] = -1

#    @jit
    def _direction_mask(self):
        """Directional Mask

        Array aligned to self.distance masking all point pairs which shall be
        ignored for binning and grouping. The one dimensional array contains
        all row-wise point pair combinations from the upper or lower triangle
        of the distance matrix in case either of both is directional.

        TODO: This array is not cached. it is used twice, for binning and
        grouping.

        Returns
        -------
        mask : numpy.array
            Array aligned to self.distance giving for each point pair
            combination a boolean value whether the point are directional or
            not.

        """
        # build the full coordinate matrix
        n = len(self._X)
        _mask = np.zeros((n, n), dtype=bool)

        # build the masking
        for i in range(n):
            loc = self.local_reference_system(poi=self._X[i])

            # apply the search radius
            sr = self._directional_model(local_ref=loc)

            _m = np.fromiter(
                (Point(p).within(sr) or Point(p).touches(sr) for p in loc),
                dtype=bool)
            _mask[:, i] = _m

        # combine lower and upper triangle
        def _indexer():
            for i in range(n):
                for j in range(n):
                    if i < j:
                        yield _mask[i, j] or _mask[j, i]

        return np.fromiter(_indexer(), dtype=bool)

    def search_area(self, poi=0, ax=None):
        """Plot Search Area

        Parameters
        ----------
        poi : integer
            Point of interest. Index of the coordinate that shall be used to
            visualize the search area.
        ax : None, matplotlib.AxesSubplot
            If not None, the Search Area will be plotted into the given
            Subplot object. If None, a new matplotlib Figure will be created
            and returned

        Returns
        -------
        plot

        """
        # get the poi
        p = self._X[poi]

        # create the local coordinate system for POI
        loc = self.local_reference_system(poi=p)

        # create the search area based on current directional model
        sa = self._directional_model(loc)

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.get_figure()

        # plot all coordinate
        ax.plot(loc[:,0], loc[:,1], '.r')

        # plot the search area
        x, y = sa.exterior.xy
        ax.plot(x, y, color='#00a562', alpha=0.7, linewidth=2,
                solid_capstyle='round')

        return fig

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

        .. code-block:: text

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

        See Also
        --------
        DirectionalVariogram._compass
        DirectionalVariogram._circle

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

    def _circle(self, local_ref):
        r"""Circular Search Area

        Construct a half-circled bounded search area for building directional
        dependent point pairs. The Search Area will be located onto the
        current point of interest and the local x-axis is rotated onto the
        azimuth angle.
        The radius of the half-circle is set to half the bandwidth.

        Parameters
        ----------
        local_ref : numpy.array
            Array of all coordinates transformed into a local representation
            with the current point of interest being the origin and the
            azimuth angle aligned onto the x-axis.

        Returns
        -------
        search_area : Polygon
            Search Area of half-circular shape bounded by the current bandwidth.

        Raises
        ------
        ValueError : In case the DirectionalVariogram.bandwidth is None or 0.

        See Also
        --------
        DirectionalVariogram._triangle
        DirectionalVariogram._compass

        """
        if self.bandwidth is None or self.bandwidth == 0:
            raise ValueError('Circular Search Area cannot be used without '
                             'bandwidth.')

        # radius is half bandwidth
        r = self.bandwidth / 2.

        # build the half circle
        circle = Point((r, 0)).buffer(r)
        half_circle = [_ for _ in circle.boundary.coords if _[0] <= r]

        # get the maximum x coordinate
        xmax = np.max(local_ref[:, 0])

        # add the bandwidth coordinates
        half_circle.extend([(xmax, r), (xmax, -r)])

        # return the figure
        return Polygon(half_circle)

    def _compass(self, local_ref):
        r"""Compass direction Search Area

        Construct a search area for building directional dependent point
        pairs. The compass search area will **not** be bounded by the
        bandwidth. It will include all point pairs at the azimuth direction
        with a given tolerance. The Search Area will be located onto the
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
            Search Area of the given compass direction.

        Notes
        -----
        The necessary figure is build by searching for the intersection of a
        half-tolerance angled line with a vertical line at the maximum x-value.
        Using polar coordinates, these points (positive and negative
        half-tolerance angle) are the edges of the search area in the local
        coordinate system. The radius of a polar coordinate can be calculated
        as:

        .. math::
            r = \frac{x}{cos (\alpha / 2)}

        The two bounding points P1 nad P2 (in local coordinates) are then
        (xmax, y) and (xmax, -y), with xmax being the maximum local
        x-coordinate representation and y:

        .. math::
            y = r * sin \left( \frac{\alpha}{2} \right)

        See Also
        --------
        DirectionalVariogram._triangle
        DirectionalVariogram._circle

        """
        # get the half tolerance angle
        a = np.radians(self.tolerance / 2)

        # get the maximum x coordinate
        xmax = np.max(local_ref[:,0])

        # calculate the radius and y coordinates
        r = xmax / a
        y = r * np.sin(a)

        # build and return the figure
        return Polygon([(0, 0), (xmax, y), (xmax, -y)])
