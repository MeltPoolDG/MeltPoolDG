#number of processes
np=${1:-4}
buildConfig=${2:"DebugRelease"}

#config dir (MeltPoolDG dependening on the location of this file)
configDir=$(dirname -- "$0")
configDir=$(realpath "$configDir")

# Get the absolute path of the script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $script_dir/log.sh

# load and apply checks
source $script_dir/check.sh
check_all

# Prompt the user to choose which dependencies to install
echo "Which dependencies do you want to install? (y/n)"
read -p "Install p4est? [y/n]: " install_p4est
read -p "Install Trilinos? [y/n]: " install_trilinos
read -p "Install ArborX? [y/n]: " install_arborx
read -p "Install deal.II? [y/n]: " install_dealii
read -p "Install adaflo? [y/n]: " install_adaflo

##############################################################
# install p4est
##############################################################
if [[ "$install_p4est" == "y" ]]; then
    VERSION=2.3.6
    wget http://p4est.github.io/release/p4est-${VERSION}.tar.gz
    mkdir -p "$(pwd)/p4est-build"
    $configDir/p4est-config.sh p4est-${VERSION}.tar.gz "$(pwd)/p4est-build"
    rm p4est-${VERSION}.tar.gz
fi

#############################################################
# install Trilinos
#############################################################
if [[ "$install_trilinos" == "y" ]]; then
    mkdir -p trilinos-build
    VERSION=16-2-0

    log "Use Trilinos version ${VERSION}."
    wget https://github.com/trilinos/Trilinos/archive/trilinos-release-${VERSION}.tar.gz
    mv trilinos-release-${VERSION}.tar.gz trilinos-release-${VERSION}.tar
    tar -xvf trilinos-release-${VERSION}.tar
    cd trilinos-build
    rm -rf *
    $configDir/trilinos-config.sh ../Trilinos-trilinos-release-${VERSION}
    rm ../trilinos-release-${VERSION}.tar

    make -j$np install
    cd ..
fi

#############################################################
# install Arborx
#############################################################
if [[ "$install_arborx" == "y" ]]; then
    VERSION=2.1
    #VERSION=1.7
    wget https://github.com/arborx/ArborX/archive/refs/tags/v${VERSION}.tar.gz
    mv v${VERSION}.tar.gz v${VERSION}.tar
    tar -xvf v${VERSION}.tar
    TRILINOS_DIR="$(pwd)/trilinos-install/lib/cmake/Kokkos"
    echo ${TRILINOS_DIR}
    ARBOX_OPTIONS=(
       -D CMAKE_INSTALL_PREFIX="./ArborX-install"
       -D ARBORX_ENABLE_MPI=ON
       -D Kokkos_DIR=${TRILINOS_DIR}
      # --debug-find-pkg=Kokkos
       )
    cmake "${ARBOX_OPTIONS[@]}" "${ARBORX_DIR:-./ArborX-${VERSION}/}"
    make install
fi

##############################################################
# install deal.II
##############################################################
if [[ "$install_dealii" == "y" ]]; then
    git clone https://github.com/dealii/dealii
    mkdir -p dealii-build
    cd dealii-build
    rm -rf *
    $configDir/dealii-config.sh ../trilinos-install ../p4est-build ../ArborX-install ../dealii $buildConfig
    make -j$np
    cd ..
fi

#############################################################
# install adaflo
#############################################################
if [[ "$install_adaflo" == "y" ]]; then
    git clone https://github.com/MeltPoolDG/adaflo
    # release
    cd adaflo
    if [[ "$buildConfig" == "Release" || "$buildConfig" == "DebugRelease" ]]; then
      mkdir -p build_release
      cd build_release
      cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Release -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
      make -j$np adaflo
      cd ..
    fi
    # debug
    if [[ "$buildConfig" == "Debug" || "$buildConfig" == "DebugRelease" ]]; then
      mkdir -p build_debug
      cd build_debug
      cmake -DBUILD_SHARED_LIBS=true -DCMAKE_BUILD_TYPE=Debug -DDEAL_II_DIR=../dealii-build -DCMAKE_CXX_COMPILER=mpicxx ..
      make -j$np adaflo
      cd ..
    fi
fi

log ""
log "Dependencies successfully installed. You may add the folders to your path via"
log "'export PATH=$(realpath $pwd/dealii-build):$PATH' >> ~/.bashrc"
log "'export ADAFLO_INCLUDE=$(realpath $pwd/adaflo/include):$PATH' >> ~/.bashrc"
