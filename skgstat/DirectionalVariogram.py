"""
Directional Variogram
"""
import numpy as np
from scipy.spatial.distance import pdist

from .Variogram import Variogram
from skgstat import plotting
from .MetricSpace import MetricSpace, MetricSpacePair


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
                 normalize=False,
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
                 **kwargs
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
            .. versionchanged:: 0.3.8
                added 'fd', 'sturges', 'scott', 'sqrt', 'doane'
            
            String identifying the binning function used to find lag class
            edges. All methods calculate bin edges on the interval [0, maxlag[.
            Possible values are:
            
                * `'even'` (default) finds `n_lags` same width bins
                * `'uniform'` forms `n_lags` bins of same data count
                * `'fd'` applies Freedman-Diaconis estimator to find `n_lags`
                * `'sturges'` applies Sturge's rule to find `n_lags`.
                * `'scott'` applies Scott's rule to find `n_lags`
                * `'doane'` applies Doane's extension to Sturge's rule to find `n_lags`
                * `'sqrt'` uses the square-root of :func:`distance <skgstat.Variogram.distance>`. as `n_lags`.

            More details are given in the documentation for :func:`set_bin_func <skgstat.Variogram.set_bin_func>`.
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
        directional_model : string, function
            The model used for selecting all points fulfilling the
            directional constraint of the Variogram. A predefined
            model can be selected by passing the model name as string.
            Optionally a callable accepting the difference vectors
            between points in polar form as angles and distances and
            returning a mask array can be passed. In this case, the
            azimuth, tolerance and bandwidth has to be incorporated by
            hand into the model.

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

        Keyword Arguments
        -----------------
        entropy_bins : int, str
            .. versionadded:: 0.3.7

            If the `estimator <skgstat.Variogram.estimator>` is set to
            `'entropy'` this argument sets the number of bins, that should be
            used for histogram calculation.
        percentile : int
            .. versionadded:: 0.3.7
            
            If the `estimator <skgstat.Variogram.estimator>` is set to
            `'entropy'` this argument sets the percentile to be used.

        """
        # Before we do anything else, make kwargs available
        self._kwargs = self._validate_kwargs(**kwargs)

        # FIXME: Call __init__ of baseclass?
        # No, because the sequence at which the arguments get initialized
        # does matter. There is way too much transitive dependence, thus
        # it was easiest to copy the init over.

        self._direction_mask_cache = None

        if not isinstance(coordinates, MetricSpace):
            coordinates = np.asarray(coordinates)
            coordinates = MetricSpace(coordinates.copy(), dist_func)
            # FIXME: Currently _direction_mask / _angles / _euclidean_dist don't get correctly calculated for sparse dspaces
            #coordinates = MetricSpace(coordinates.copy(), dist_func, maxlag if maxlag and not isinstance(maxlag, str) and maxlag >= 1 else None)
        else:
            assert self.dist_func == coordinates.dist_metric, "Distance metric of variogram differs from distance metric of coordinates"
            assert coordinates.max_dist is None
            
        # Set coordinates
        self._X = coordinates

        # pairwise difference
        self._diff = None

        # set verbosity
        self.verbose = verbose

        # set values
        self._values = None
        # calc_diff = False here, because it will be calculated by fit() later
        self.set_values(values=values, calc_diff=False)
        
        # distance matrix
        self._dist = None

        # set distance calculation function
        self._dist_func_name = None
        self.set_dist_function(func=dist_func)

        # Angles and euclidean distances used for direction mask calculation
        self._angles = None
        self._euclidean_dist = None

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
        # Note that fit() calls preprocessing
        self.fit(force=True)

    def preprocessing(self, force=False):
        self._calc_direction_mask_data(force)
        self._calc_diff(force=force)
        self._calc_groups(force=force)

    def _calc_direction_mask_data(self, force=False):
        r"""
        Calculate directional mask data.
        For this, the angle between the vector between the two
        points, and east (see comment about self.azimuth) is calculated.
        The result is stored in self._angles and contains the angle of each
        point pair vector to the x-axis in radians.

        Parameters
        ----------
        force : bool
            If True, a new calculation of all angles is forced, even if they
            are already in the cache.

        Notes
        -----
        The masked data is in radias, while azimuth is given in degrees.
        For the Vector between a point pair A,B :math:`\overrightarrow{AB}=u` and the
        x-axis, represented by vector :math:`\overrightarrow{e} = [1,0]`, the angle
        :math:`\Theta` is calculated like:

        .. math::
            cos(\Theta) = \frac{u \circ e}{|e| \cdot |[1,0]|}

        See Also
        --------
        `azimuth <skgstat.DirectionalVariogram.azimuth>`_

        """

        # FIXME: This should be optimized for the sparse case (range << bbox(coordinates)),
        # i.e. use the MetricSpace in self._X
        
        # check if already calculated
        if self._angles is not None and not force:
            return

        # if self.coordinates is of just one dimension, concat zeros.
        if self.coordinates.ndim == 1:
            _x = np.vstack(zip(self.coordinates, np.zeros(len(self.coordinates))))
        elif self.coordinates.ndim == 2:
            _x = self.coordinates
        else:
            raise NotImplementedError('N-dimensional coordinates cannot be handled')

        # for angles, we need Euklidean distance,
        # no matter which distance function is used
        #if self._dist_func_name == "euclidean":
        #    self._euclidean_dist = scipy.spatial.distance.squareform(self.distance_matrix)
        #else:
        self._euclidean_dist = pdist(_x, "euclidean")

        # Calculate the angles
        # (a - b).[1,0] = ||a - b|| * ||[1,0]|| * cos(v)
        # cos(v) = (a - b).[1,0] / ||a - b||
        # cos(v) = (a.[1,0] - b.[1,0]) / ||a - b||
        scalar = pdist(np.array([np.dot(_x, [1, 0])]).T, np.subtract)
        pos_angles = np.arccos(scalar / self._euclidean_dist)

        # cos(v) for [2,1] and [2, -1] is the same,
        # but v is not (v vs -v), fix that.
        ydiff = pdist(np.array([np.dot(_x, [0, 1])]).T, np.subtract)

        # store the angle or negative angle, depending on the
        # amount of the x coordinate
        self._angles = np.where(ydiff >= 0, pos_angles, -pos_angles)


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

        # reset groups and mask cache on azimuth change
        self._direction_mask_cache = None
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

        # reset groups and mask on tolerance change
        self._direction_mask_cache = None
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

        # reset groups and direction mask cache on bandwidth change
        self._direction_mask_cache = None
        self._groups = None

    def set_directional_model(self, model_name):
        """Set new directional model

        The model used for selecting all points fulfilling the
        directional constraint of the Variogram. A predefined model
        can be selected by passing the model name as string.
        Optionally a callable accepting the difference vectors between
        points in polar form as angles and distances and returning a
        mask array can be passed. In this case, the azimuth, tolerance
        and bandwidth has to be incorporated by hand into the model.
        The predefined options are:

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
            The name of the predefined model (string) or a function
            that accepts angle and distance arrays and returns a mask
            array.

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

    @property
    def bins(self):
        if self._bins is None:
            # get the distances
            d = self.distance.copy()
            d[np.where(~self._direction_mask())] = np.nan

            self._bins, n = self.bin_func(d, self._n_lags, self.maxlag)

            # if the binning function returned an N, the n_lags need
            # to be adjusted directly (not through the setter)
            if n is not None:
                self._n_lags = n

        return self._bins.copy()

    def to_gstools(self, *args, **kwargs):
        raise NotImplementedError(
            "Exporting DirectinalVariogram is currently not supported."
        )

    def _calc_groups(self, force=False):
        super(DirectionalVariogram, self)._calc_groups(force=force)

        # set to outside maxlag group
        self._groups[np.where(~self._direction_mask())] = -1

#    @jit
    def _direction_mask(self, force=False):
        """Directional Mask

        Array aligned to self.distance masking all point pairs which shall be
        ignored for binning and grouping. The one dimensional array contains
        all row-wise point pair combinations from the upper or lower triangle
        of the distance matrix in case either of both is directional.

        Returns
        -------
        mask : numpy.array
            Array aligned to self.distance giving for each point pair
            combination a boolean value whether the point are directional or
            not.

        """

        if force or self._direction_mask_cache is None:
            self._direction_mask_cache = self._directional_model(self._angles, self._euclidean_dist)
        return self._direction_mask_cache

    def pair_field(self, ax=None, cmap="gist_rainbow", points='all', add_points=True, alpha=0.3, **kwargs):  # pragma: no cover
        """
        Plot a pair field.

        Plot a network graph for all point pairs that fulfill the direction
        filter and lie within each others search area.

        Parameters
        ----------
        ax : matplotlib.Subplot
            A matplotlib Axes object to plot the pair field onto.
            If ``None``, a new new matplotlib figure will be created.
        cmap : string
            Any color-map name that is supported by matplotlib
        points : 'all', int, list
            If not ``'all'``, only the given coordinate (int) or 
            list of coordinates (list) will be plotted. Recommended, if
            the input data is quite large.
        add_points : bool
            If True (default) The coordinates will be added as black points.
        alpha : float
            Alpha value for the colors to make overlapping vertices
            visualize better. Defaults to ``0.3``.
            
        """
        # get the backend
        used_backend = plotting.backend()

        if used_backend == 'matplotlib':
            return plotting.matplotlib_pair_field(self, ax=ax, cmap=cmap, points=points, add_points=add_points, alpha=alpha, **kwargs)
        elif used_backend == 'plotly':
            return plotting.plotly_pair_field(self, fig=ax, points=points, add_points=add_points, alpha=alpha, **kwargs)       

    def _triangle(self, angles, dists):
        r"""Triangular Search Area

        Construct a triangular bounded search area for building directional
        dependent point pairs. The Search Area will be located onto the
        current point of interest and the local x-axis is rotated onto the
        azimuth angle.

        Parameters
        ----------
        angles, dists : numpy.array
            Vectors between point pairs in polar form (angle relative
            to east in radians, length in coordinate space units)

        Returns
        -------
        mask : numpy.array(bool)
            Point pair mask, indexed as the results of
            scipy.spatial.distance.pdist are.

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

        absdiff = np.abs(angles + np.radians(self.azimuth))
        absdiff = np.where(absdiff > np.pi, absdiff - np.pi, absdiff)
        absdiff = np.where(absdiff > np.pi / 2, np.pi - absdiff, absdiff)

        in_tol = absdiff <= np.radians(self.tolerance / 2)
        in_band = self.bandwidth / 2 >= np.abs(dists * np.sin(np.abs(angles + np.radians(self.azimuth))))

        return in_tol & in_band

    def _circle(self, angles, dists):
        r"""Circular Search Area

        Construct a half-circled bounded search area for building directional
        dependent point pairs. The Search Area will be located onto the
        current point of interest and the local x-axis is rotated onto the
        azimuth angle.
        The radius of the half-circle is set to half the bandwidth.

        Parameters
        ----------
        angles, dists : numpy.array
            Vectors between point pairs in polar form (angle relative
            to east in radians, length in coordinate space units)

        Returns
        -------
        mask : numpy.array(bool)
            Point pair mask, indexed as the results of
            scipy.spatial.distance.pdist are.

        Raises
        ------
        ValueError : In case the DirectionalVariogram.bandwidth is None or 0.

        See Also
        --------
        DirectionalVariogram._triangle
        DirectionalVariogram._compass

        """
        raise NotImplementedError

    def _compass(self, angles, dists):
        r"""Compass direction direction mask

        Construct a search area for building directional dependent point
        pairs. The compass search area will **not** be bounded by the
        bandwidth. It will include all point pairs at the azimuth direction
        with a given tolerance. The Search Area will be located onto the
        current point of interest and the local x-axis is rotated onto the
        azimuth angle.

        Parameters
        ----------
        angles, dists : numpy.array
            Vectors between point pairs in polar form (angle relative
            to east in radians, length in coordinate space units)

        Returns
        -------
        mask : numpy.array(bool)
            Point pair mask, indexed as the results of
            scipy.spatial.distance.pdist are.

        See Also
        --------
        DirectionalVariogram._triangle
        DirectionalVariogram._circle

        """

        absdiff = np.abs(angles + np.radians(self.azimuth))
        absdiff = np.where(absdiff > np.pi, absdiff - np.pi, absdiff)
        absdiff = np.where(absdiff > np.pi / 2, np.pi - absdiff, absdiff)

        return absdiff <= np.radians(self.tolerance / 2)
