# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from importlib.metadata import version as get_version

from packaging.version import Version

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "thuban"
copyright = "2024, PUNCH Science Operations Center"
author = "PUNCH Science Operations Center"

release: str = get_version("thuban")
version: str = release
_version = Version(release)
if _version.is_devrelease:
    version = release = f"{_version.base_version}.dev{_version.dev}"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension",
              "sphinx.ext.autodoc",
              "sphinx.ext.napoleon",
              "sphinx_favicon",
              "IPython.sphinxext.ipython_console_highlighting"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/punch-mission/thuban",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
    "logo": {
        "text": "thuban",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    }
}
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "punch-mission",
    "github_repo": "thuban",
    "github_version": "main",
    "doc_path": "docs/",
}


autoapi_dirs = ["../thuban"]

favicons = ["favicon.ico"]
