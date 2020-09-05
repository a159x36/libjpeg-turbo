IMAGE=../testimages/Sunrise.bmp
BUILD_DIR=../newbuild
HALIDE_BIN_PATH ?= /usr/local/
HALIDE_SRC_PATH ?= /usr/local/share/halide
CXXFLAGS += -std=c++11 -I $(HALIDE_BIN_PATH)/include/ -I $(HALIDE_SRC_PATH)/tools/ -I $(HALIDE_SRC_PATH)/apps/support/

LIB_HALIDE = $(HALIDE_BIN_PATH)/lib/libHalide.a
LLVM_CONFIG ?= llvm-config
LLVM_LIBDIR = $(shell $(LLVM_CONFIG) --libdir)
LLVM_FULL_VERSION = $(shell $(LLVM_CONFIG) --version)
LLVM_SHARED_LIBS = -L $(LLVM_LIBDIR) 
GENERATOR_SRC=$(HALIDE_SRC_PATH)/tools/GenGen.cpp
CXX = clang++
all: idct_slow idct_fast
%:: %_generator.cpp
	$(CXX) $(CXXFLAGS) $< $(GENERATOR_SRC) -g $(LIB_HALIDE) -o $@ -lpthread -fno-rtti -ldl -lz -lncurses -rdynamic $(LDFLAGS) $(LLVM_SHARED_LIBS)
assembly: idct_fast idct_slow x86_64/idct_fast.s x86_64/idct_slow.s
armasm: idct_fast idct_slow armv7l/idct_fast.s armv7l/idct_slow.s
GENERATOR_OPTS=-no_asserts-no_bounds_query
EMIT=-e o,h
define haliderules
$(1)/%.o: %
	./$$< -g $$< -o $(1) target=$(2)$(GENERATOR_OPTS) $(EMIT) auto_schedule=false
$(1)/%.s: %
	./$$< -g $$< -o $(1) target=$(2)$(GENERATOR_OPTS)-no_runtime -e assembly,stmt
endef
$(eval $(call haliderules,x86_64,host))
$(eval $(call haliderules,aarch64,arm-64))
$(eval $(call haliderules,armv7l,arm-32))
HALIDE_OBJS=x86_64/idct_fast.o x86_64/idct_slow.o x86_64/idct4_slow.o
HALIDE_GENERATORS=idct_fast idct_slow
.SECONDARY: $(HALIDE_GENERATORS)
TJBENCHOPTS=-quiet  -benchtime 0.001 -warmup 0.01 -nowrite
tjbench:
	make -C $(BUILD_DIR) -j 8
test420:  $(HALIDE_OBJS) tjbench
	$(BUILD_DIR)/tjbench $(IMAGE) 95 -subsamp 420 $(TJBENCHOPTS) -fastdct -halide
test: $(HALIDE_OBJS) tjbench
	$(BUILD_DIR)/tjbench $(IMAGE) 95 -subsamp GRAY $(TJBENCHOPTS) -accuratedct
	$(BUILD_DIR)/tjbench $(IMAGE) 95 -subsamp GRAY $(TJBENCHOPTS) -fastdct
	$(BUILD_DIR)/tjbench $(IMAGE)  95 -subsamp GRAY $(TJBENCHOPTS) -accuratedct -halide
	$(BUILD_DIR)/tjbench $(IMAGE) 95 -subsamp GRAY $(TJBENCHOPTS) -fastdct -halide
clean:
	rm */*.o $(HALIDE_GENERATORS)
