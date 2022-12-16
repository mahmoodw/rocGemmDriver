BLIS = -fopenmp -lblis -L extern/blis/lib/ -rpath extern/blis/lib/
FLAME = -lflame -L extern/flame/lib/ -rpath extern/flame/lib/
BOOST = -lboost_program_options

# CFLAGS=-DHIPBLAS=1
# CPPFLAGS += -DHIPBLAS
ifeq ($(BUILD_WITH_HIPBLAS),1)
CFLAGS=-DHIPBLAS=1 
ifeq ($(HIPBLASPATH),)
HIPBLASLIB = -lhipblas -L /opt/rocm/hipblas/lib/
HIPBLASINCL = -I /opt/rocm/hipblas/include/
else
ifeq ($(DEBUG),1)
HIPBLASLIB = -lhipblas -L $(HIPBLASPATH)/build/debug/library/src/ --linker-options=-rpath,$(HIPBLASPATH)/build/debug/library/src/
HIPBLASINCL = -I $(HIPBLASPATH)/library/include/ -I $(HIPBLASPATH)/build/debug/include/ -I $(HIPBLASPATH)/build/debug/include/internal
else
HIPBLASLIB = -lhipblas -L $(HIPBLASPATH)/build/release/library/src/ --linker-options=-rpath,$(HIPBLASPATH)/build/release/library/src/
HIPBLASINCL = -I $(HIPBLASPATH)/library/include/ -I $(HIPBLASPATH)/build/release/include/ -I $(HIPBLASPATH)/build/release/include/internal
endif
endif
endif

ifneq ($(BUILD_WITH_HIPBLAS),1)
ifeq ($(ROCBLASPATH),)
ROCBLASLIB = -lrocblas -L /opt/rocm/rocblas/lib/
ROCBLASINCL = -I /opt/rocm/rocblas/include/
else
ifeq ($(DEBUG),1)
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/debug/library/src/ -rpath $(ROCBLASPATH)/build/debug/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/debug/include/ -I $(ROCBLASPATH)/build/debug/include/internal
else
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/release/library/src/ -rpath $(ROCBLASPATH)/build/release/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/release/include/ -I $(ROCBLASPATH)/build/release/include/internal
endif
endif
endif

ifeq ($(CLANG),1)
LINKER_FLAG = -rtlib=compiler-rt
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/hip/bin/hipcc -g -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(HIPBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/hip/bin/hipcc -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(HIPBLASLIB) $(BOOST)
endif
else
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o
	/opt/rocm/hip/bin/hipcc -g -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(HIPBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o
	/opt/rocm/hip/bin/hipcc -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(HIPBLASLIB) $(BOOST) 
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/hip/bin/hipcc -g -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) $(HIPBLASINCL) $(CFLAGS) -I extern/flame/include/ -DVALIDATE
else
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/hip/bin/hipcc -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) $(HIPBLASINCL) $(CFLAGS) -I extern/flame/include/ -DVALIDATE
endif
else
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/hip/bin/hipcc -g -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) $(HIPBLASINCL) $(CFLAGS)
else
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/hip/bin/hipcc -c GemmDriver.cpp -std=c++14 $(ROCBLASINCL) $(HIPBLASINCL) $(CFLAGS)
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/hip/bin/hipcc -g -c blis_interface.cpp -std=c++14 -I extern/blis/include/blis/ $(ROCBLASINCL) $(HIPBLASINCL)
else
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/hip/bin/hipcc -c blis_interface.cpp -std=c++14 -I extern/blis/include/blis/ $(ROCBLASINCL) $(HIPBLASINCL)
endif
endif

clean:
	rm -f GemmDriver *.o
