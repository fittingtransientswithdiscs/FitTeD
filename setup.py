from setuptools import setup
import codecs	
import os.path

exts =[] 

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")	
	
	
if __name__ == "__main__" :
	setup(
	   name='FitTeD',
	   version=get_version("fitted/__init__.py"),
	   author='Andrew Mummery',
	   author_email='andrew.mummery@physics.ox.ac.uk',
	   packages=['fitted', ],
	   description='A pacakged for fitting Astrophysical transients with disc models.',
	   long_description=open('README.md').read(),
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
           "tqdm"],
	   ext_modules = exts,
	   python_requires='>=3.5, <4'
	)
