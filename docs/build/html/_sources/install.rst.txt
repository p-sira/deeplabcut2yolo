..
   deeplabcut2yolo is licensed under GNU General Public License v3.0, see LICENSE.
   Copyright 2024 Sira Pornsiriprasert <code@psira.me>


Installation
============

To install deeplabcut2yolo using ``pip``:

.. code-block:: console

    pip install deeplabcut2yolo


Or build the module yourself. Make sure you have python-build installed, if not:

.. code-block:: console

    sudo pacman -Syu python-build

For python-build installation on your specific system, please refer to your favorite package manager's manual.
Once, python-build is installed:

.. code-block:: console

    git clone https://github.com/p-sira/deeplabcut2yolo/
    cd deeplabcut2yolo
    python -m build
    pip install .