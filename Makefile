BLIS = -fopenmp -lblis -L extern/blis/lib/ -rpath extern/blis/lib/
FLAME = -lflame -L extern/flame/lib/ -rpath extern/flame/lib/
BOOST = -lboost_program_options

CFLAGS=-DROCM_USE_FLOAT16=1 

ifeq ($(ROCBLASPATH),)
ROCBLASLIB = -lrocblas -L /opt/rocm/rocblas/lib/
ROCBLASINCL = -I /opt/rocm/include/rocblas
else
ifeq ($(DEBUG),1)
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/debug/library/src/ -rpath $(ROCBLASPATH)/build/debug/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/debug/include/rocblas/ -I $(ROCBLASPATH)/build/debug/include/rocblas/internal/
else
ROCBLASLIB = -lrocblas -L $(ROCBLASPATH)/build/release/library/src/ -rpath $(ROCBLASPATH)/build/release/library/src/
ROCBLASINCL = -I $(ROCBLASPATH)/library/include/ -I $(ROCBLASPATH)/build/release/include/rocblas/ -I $(ROCBLASPATH)/build/release/include/rocblas/internal/ 
endif
endif

ifeq ($(CLANG),1)
LINKER_FLAG = -rtlib=compiler-rt
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/bin/hipcc -g -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o blis_interface.o
	/opt/rocm/bin/hipcc -o GemmDriver GemmDriver.o blis_interface.o $(LINKER_FLAG) $(BLIS) $(FLAME)  $(ROCBLASLIB) $(BOOST)
endif
else
ifeq ($(DEBUG),1)
GemmDriver: GemmDriver.o
	/opt/rocm/bin/hipcc -g -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(BOOST)
else
GemmDriver: GemmDriver.o
	/opt/rocm/bin/hipcc -o GemmDriver GemmDriver.o $(LINKER_FLAG) $(ROCBLASLIB) $(BOOST)
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/bin/hipcc -g -c GemmDriver.cpp -std=c++17 $(CFLAGS) $(ROCBLASINCL) -I extern/flame/include/ -DVALIDATE
else
GemmDriver.o: GemmDriver.cpp utility.hpp validate.hpp flame_interface.hpp
	/opt/rocm/bin/hipcc -c GemmDriver.cpp -std=c++17 $(CFLAGS) $(ROCBLASINCL) -I extern/flame/include/ -DVALIDATE
endif
else
ifeq ($(DEBUG),1)
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/bin/hipcc -g -c GemmDriver.cpp -std=c++17 $(CFLAGS) $(ROCBLASINCL) 
else
GemmDriver.o: GemmDriver.cpp utility.hpp
	/opt/rocm/bin/hipcc -c GemmDriver.cpp -std=c++17 $(CFLAGS) $(ROCBLASINCL) 
endif
endif

ifeq ($(VALIDATE),1)
ifeq ($(DEBUG),1)
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/bin/hipcc -g -c blis_interface.cpp -std=c++17 -I extern/blis/include/blis/ $(CFLAGS) $(ROCBLASINCL)
else
blis_interface.o: blis_interface.cpp blis_interface.hpp utility.hpp
	/opt/rocm/bin/hipcc -c blis_interface.cpp -std=c++17 -I extern/blis/include/blis/ $(CFLAGS) $(ROCBLASINCL)
endif
endif

clean:
	rm -f GemmDriver *.o
