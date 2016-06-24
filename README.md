# Installation Instruction for Mac OS X

## Install Anaconda

1. Download Anaconda (it is very well packaged, includes everything you need for jupyter, even python comes with it): http://repo.continuum.io/archive/Anaconda2-4.0.0-MacOSX-x86_64.sh (Python3 version [here](http://repo.continuum.io/archive/Anaconda3-4.0.0-MacOSX-x86_64.sh))

2. Once the script is downloaded, install in terminal with command `bash /path/to/Anaconda2-4.0.0-MacOSX-x86_64.sh`. Depending on the shell you are using, you might need to update your own PATH environment variable to include anaconda's `bin` folder.

3. After installing and restarting shell, you should be able to open the notebook web page by running `jupyter notebook`. You can use the Web UI to navigate to the notebook directory and open within.

## Install GraphLab Create via pip

Run the following command

    conda update pip
    pip install graphlab-create --no-cache-dir
