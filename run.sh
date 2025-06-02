# This is necessary to give path to compiler
# If this doesn't work, just put the commands in terminal one by one
export PATH="/c/msys64/ucrt64/bin:$PATH"
export CC=/c/msys64/ucrt64/bin/gcc.exe
export CXX=/c/msys64/ucrt64/bin/g++.exe

# Remove build folder if exists
if [ -d build ]; then
    echo "Remove old build folder"
    #rm -rf build
else
    echo "build doesn't exist"
fi

# Remove logs file if exists
if [ -f clang-tidy-report.log ]; then
    echo "Remove old log file"
    rm clang-tidy-report.log
else
    echo "log file doesn't exist"
fi

# Configurate project
cmake -S . -B build

# Build the project
cmake --build build 

# Apply formatting to all files
cmake --build build --target format 

# Apply code analysis to all files and save it in clang-tidy-report.log
cmake --build build --target tidy

# Run tests
./build/tests