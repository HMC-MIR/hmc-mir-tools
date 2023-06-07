def myversion_func(version: ScmVersion):
    from setuptools_scm.version import guess_next_version
    print('myversion_func')
    print(version)
    return version.format_next_version(guess_next_version, '{guessed}b{distance}')