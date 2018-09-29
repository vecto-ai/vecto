"""Setup script for vecto package."""

import setup_boilerplate


class Package(setup_boilerplate.Package):

    """Package metadata."""

    name = 'vecto'
    description = 'toolbox for various tasks in the area of vector space models of computational linguistic'
    url = "http://vecto.space"
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: POSIX',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Text Processing :: Linguistic']
    keywords = ['NLP', 'linguistics', 'language']


if __name__ == '__main__':
    Package.setup()
