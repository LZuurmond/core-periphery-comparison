# --- Configuration ---
# Compiler to use
CXX = g++

# C++ Standard
CXXFLAGS = -std=c++17 -Wall -Wextra -O3

# Include directories for GSL (often found in /usr/include or /usr/local/include)
# Adjust these paths if GSL is installed in a non-standard location
GSL_INCLUDE =

# Linker flags for GSL library (including gsl and gslcblas)
LDFLAGS = -lgsl -lgslcblas -lm

# --- Source and Object Files ---
# All source files required for the executable
SRCS = main.cpp hierarchical_model.cpp readgml.cpp parameters.cpp indexed_list.cpp

# Corresponding object files
OBJS = $(SRCS:.cpp=.o)

# Name of the final executable
TARGET = hcp

# --- Targets ---

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

# Rule for compiling C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(GSL_INCLUDE) -c $< -o $@

# Clean target to remove generated files
clean:
	rm -f $(OBJS) $(TARGET)