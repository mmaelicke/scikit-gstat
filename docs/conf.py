# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('.'))


def get_version():
    B = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(B, '..', 'skgstat', '__version__.py'), 'r') as f:
        loc = dict()
        exec(f.read(), loc, loc)
        return loc['__version__']

# -- Project information -----------------------------------------------------


project = 'SciKit GStat'
copyright = '2022, Mirko Mälicke'
author = 'Mirko Mälicke'

# The short X.Y version
# version = '0.3.2'
# The full version, including alpha/beta/rc tags
release = get_version()


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# RST settings
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: math(raw)
   :format: latex html

.. default-role:: code

.. |br| raw:: html

   <br />

.. |nbsp| unicode:: 0xA0
   :trim:
"""

# Add any Sphinx extension module names here, as strings.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
    'sphinx_gallery.gen_gallery',
    'sphinx.ext.imgmath',
    'sphinx.ext.imgconverter',
    'sphinx.ext.autosectionlabel',
]

# IPython directive configuration
ipython_warning_is_error = False  # Don't fail on warnings
ipython_execlines = []  # No default imports
ipython_holdcount = True  # Maintain execution count with @suppress

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ['.rst', '.ipynb']

# The master toctree document.
master_doc = 'index'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# -- Options for docutils ----------------------------------------------
docutils_tab_width = 4
trim_doctest_flags = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'pydata_sphinx_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    'navigation_depth': 4,
    'show_prev_next': False,
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/mmaelicke/scikit-gstat',
            'icon': 'fab fa-github-square',
        },
    ],
}

html_context = {
    'github_user': 'mmaelicke',
    'github_repo': 'scikit-gstat',
    'github_version': 'master',
    'doc_path': 'docs'
}

html_short_title = 'SciKit-GStat'
"""
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
        'donate.html'
    ]
}
"""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'SciKitGStatdoc'


# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'SciKitGStat.tex', 'SciKit GStat Documentation',
     'Mirko Mälicke', 'manual'),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'scikitgstat', 'SciKit GStat Documentation',
     [author], 1)
]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'SciKitGStat', 'SciKit GStat Documentation',
     author, 'SciKitGStat', 'One line description of project.',
     'Miscellaneous'),
]


# -- Extension configuration -------------------------------------------------

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {'https://docs.python.org/': None}

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.6', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'numpy':  ('https://docs.scipy.org/doc/numpy', None),
    'scipy':  ('https://docs.scipy.org/doc/scipy/reference', None),
    'gstools': ('https://geostat-framework.readthedocs.io/projects/gstools/en/latest/', None),
    'sklearn': ('http://scikit-learn.org/stable', None),
}

from plotly.io._sg_scraper import plotly_sg_scraper
image_scrapers = ('matplotlib', plotly_sg_scraper,)
import plotly.io as pio
pio.renderers.default = 'sphinx_gallery'

import sphinx_gallery

# Configure sphinx-gallery
sphinx_gallery_conf = {
    'examples_dirs': ['../tutorials'],  # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('skgstat', 'skgstat'),
    'image_scrapers': image_scrapers,
    'filename_pattern': '.*',  # Include all files
    'ignore_pattern': r'\.py\.md5$|\.codeobj\.json$|\.zip$|\.rst$|\.ipynb$|\.py\.md5|\.py\.ipynb|\.py\.py|\.pickle$',
    'notebook_images': True,
    'remove_config_comments': True,
    'download_all_examples': True,
    'within_subsection_order': lambda x: os.path.basename(x),  # Order by filename
    'capture_repr': ('_repr_html_', '__repr__'),
    'first_notebook_cell': None,
    'thumbnail_size': (400, 400),
    'min_reported_time': 0,
    'show_memory': False,
    'junit': None,
    'plot_gallery': True,
    'reset_modules': ('matplotlib', 'seaborn'),
    'reference_url': {
        'skgstat': None,
        'numpy': 'https://numpy.org/doc/stable',
        'scipy': 'https://docs.scipy.org/doc/scipy/reference'
    },
    'binder': {
        'org': 'mmaelicke',
        'repo': 'scikit-gstat',
        'branch': 'master',
        'binderhub_url': 'https://mybinder.org',
        'dependencies': '../requirements.txt',  # Point to the root directory
    }
}

# Configure math options
imgmath_image_format = 'svg'
imgmath_font_size = 14

# Configure RST settings
rst_prolog = """
.. role:: python(code)
   :language: python
   :class: highlight

.. role:: math(raw)
   :format: latex html

.. default-role:: code

.. |br| raw:: html

   <br />

.. |nbsp| unicode:: 0xA0
   :trim:
"""

# Configure math settings
mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-AMS_HTML'
mathjax_config = {
    'tex2jax': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
        'processEscapes': True,
        'processEnvironments': True
    }
}

# Configure autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Configure napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Configure docutils settings
docutils_settings = {
    'file_insertion_enabled': True,
    'raw_enabled': True,
    'report_level': 2,
    'strip_comments': True,
    'strip_elements_with_classes': [],
    'strip_classes': [],
    'initial_header_level': 2,
    'warning_stream': None,
    'embed_stylesheet': False,
    'cloak_email_addresses': True,
    'pep_base_url': 'https://www.python.org/dev/peps/',
    'pep_references': None,
    'rfc_base_url': 'https://tools.ietf.org/html/',
    'rfc_references': None,
    'input_encoding': 'utf-8',
    'doctitle_xform': True,
    'sectsubtitle_xform': False,
    'section_self_link': False,
    'footnote_references': 'superscript',
    'trim_footnote_reference_space': True,
    'smart_quotes': True,
    'language_code': 'en',
    'syntax_highlight': 'long'
}
