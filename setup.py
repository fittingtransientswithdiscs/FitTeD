from setuptools import setup, find_packages
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), 'r', encoding='utf8') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")


if __name__ == "__main__":
    setup(
        name='FitTeD',
        version=get_version("fitted/__init__.py"),
        author='Andrew Mummery',
        author_email='amummery@ias.edu',
        description=('Fitting astrophysical transients with relativistic '
                     'accretion disc models.'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        url='https://github.com/fittingtransientswithdiscs/FitTeD',
        license='BSD-3-Clause',

        # find_packages, NOT a hand-written list.  The previous list named
        # only the top-level package, which under meson-python did not
        # matter (meson installed the tree) but under setuptools would ship
        # a package with no models/ subpackage at all.
        packages=find_packages(include=['fitted', 'fitted.*']),

        # Data files that must travel with the code.  examples/ and
        # models/tests/ have no __init__.py, so they are data, not packages.
        package_data={
            'fitted': ['fitted_style.mplstyle', 'examples/*.txt'],
            'fitted.models': ['*.npy', 'tests/*.py', 'tests/*.npz'],
            'fitted.models.raytrace': ['*.md'],
        },

        install_requires=[
            "emcee",
            "numpy",
            "scipy",
            "pandas",
            "astropy",
            "matplotlib",
            "importlib_resources",
            "corner",
            "h5py",
            "tqdm",
            # GR_disc defaults to use_iv_approximation=True, so numba is on
            # the default path.  Without it the code silently falls back to
            # scipy.special.iv and the numbers move slightly, which would
            # make the published tutorial outputs unreproducible.
            "numba",
        ],
        python_requires='>=3.10',
        classifiers=[
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3 :: Only",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "License :: OSI Approved :: BSD License",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Astronomy",
            "Operating System :: OS Independent",
        ],
    )
