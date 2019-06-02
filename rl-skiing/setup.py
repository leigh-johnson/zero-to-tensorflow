#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
import subprocess
from setuptools import setup, find_packages, Command
from distutils.command.build import build as _build


requirements = ['cmake', 'gym[atari]', 'tensorflow;sys_platform=="darwin"',
                'tensorflow;sys_platform=="linux"',
                'numpy', 'agents']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

_LIBS = ('python-numpy python-dev cmake zlib1g-dev').split()

CUSTOM_COMMANDS = [
    ['apt-get', 'update'],
    ['apt-get', 'install', '-y'] + _LIBS,
]


class CustomCommands(Command):
    """A setuptools Command class able to run arbitrary commands."""

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print('Running command: %s' % command_list)
        p = subprocess.Popen(
            command_list,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Can use communicate(input='y\n'.encode()) if the command run requires
        # some confirmation.
        stdout_data, _ = p.communicate()
        print('Command output: %s' % stdout_data)
        if p.returncode != 0:
            raise RuntimeError(
                'Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)


class build(_build):
    """A build command class that will be invoked during package install.
      The package built using the current setup.py will be staged and later
      installed in the worker using `pip install package'. This class will be
      instantiated during install for this specific scenario and will trigger
      running the custom commands specified.
      """
    sub_commands = _build.sub_commands + [('CustomCommands', None)]


setup(
    author="Leigh Johnson",
    author_email='leigh@data-literate.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description="From Zero to Reinforcement Learning: training an Atari bot using Tensorflow, Keras, OpenAI Gym",
    install_requires=requirements,
    license="MIT license",
    include_package_data=True,
    keywords='zero_to_rl_skiing',
    name='zero_to_rl_skiing',
    packages=find_packages(include=['trainer']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/leigh-johnson/zero_to_rl_skiing',
    version='0.0.1',
    zip_safe=False,
    cmdclass={
        # Command class instantiated and run during pip install scenarios.
        'build': build,
        'CustomCommands': CustomCommands,
    }
)
