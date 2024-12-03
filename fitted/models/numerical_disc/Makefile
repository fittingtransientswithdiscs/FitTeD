src=	amodules.f90              \
		tdedisc_grid.pyf          \
		tdedisc_grid.f90          \

make:
	python3 -m numpy.f2py -c $(src) 

clean:
	rm *.so