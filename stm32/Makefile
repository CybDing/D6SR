######################################################################
#  球形机器人STM32项目Makefile
######################################################################

# 项目名称和源文件配置
BINARY		= spherical_robot

# 新框架源文件 (统一PID控制)
SRCFILES	= main.c uart_comm.c motor_control.c imu_reader.c pid_control.c \
		  axis_stabilizer.c motor_calibration.c motion_controller.c \
		  cascaded_pid.c

# 旧框架源文件 (保留用于回退,当前未编译)
# SRCFILES	+= roll_stabilizer.c pitch_stabilizer.c

# FreeRTOS核心源文件路径
FREERTOS_KERNEL = ../rtos/FreeRTOS-Kernel

SRCFILES	+= $(FREERTOS_KERNEL)/tasks.c
SRCFILES	+= $(FREERTOS_KERNEL)/queue.c
SRCFILES	+= $(FREERTOS_KERNEL)/list.c
# SRCFILES 	+= $(FREERTOS_KERNEL)/timer.c

# 目标硬件: STM32F407VET6
# FreeRTOS移植层
FREERTOS_PORT = $(FREERTOS_KERNEL)/portable/GCC/ARM_CM4F
LDSCRIPT	= stm32f407vet6.ld
SRCFILES	+= $(FREERTOS_PORT)/port.c

# FreeRTOS内存管理
SRCFILES	+= $(FREERTOS_KERNEL)/portable/MemMang/heap_4.c

# 包含主要的编译配置文件
include ../Makefile.incl
include ../rtos/Makefile.rtos

######################################################################
# Makefile参数说明:
#
# BINARY      = 生成的可执行文件名称
# SRCFILES    = 所有需要编译的C源文件列表
# LDSCRIPT    = 链接脚本文件，定义STM32内存布局
#
# FreeRTOS核心文件:
# tasks.c     = FreeRTOS任务管理核心
# queue.c     = 队列通信机制
# list.c      = 链表数据结构实现
# timers.c    = 软件定时器
# port.c      = ARM Cortex-M4F移植层
# heap_4.c    = 内存管理(支持内存碎片整理)
#
# 编译标志在Makefile.incl中定义:
# -DSTM32F4              = STM32F4系列宏定义
# -mthumb -mcpu=cortex-m4 = ARM Cortex-M4F架构(硬件浮点)
# -Os -g                  = 大小优化+调试信息
######################################################################

######################################################################
#  使用说明:
#	编译项目:       make
#	生成bin文件:    make bin  
#	清理对象文件:   make clean
#	完全清理:       make clobber
#	烧录程序:       make flash
#	查看配置:       make info
#	串口监控:       make monitor
######################################################################

# 自定义目标: 监控串口输出
monitor:
	@echo "串口监控 (退出: Ctrl+A X)"
	minicom -D /dev/ttyUSB0 -b 115200

# 自定义目标: 使用screen监控串口
screen:
	@echo "串口监控 (退出: Ctrl+A K)"
	screen /dev/ttyUSB0 115200

# 自定义目标: 全量重新编译 (清理 -> 编译)
f407:
	$(MAKE) clobber
	$(MAKE)

# 自定义目标: 完整编译 + 烧录 (清理 -> 编译 -> 写入 Flash)
f407-program:
	$(MAKE) clobber
	$(MAKE) bin
	$(MAKE) flash

# 自定义目标: 增量编译并烧录
program: bin flash

# 显示编译配置信息
info:
	@echo "球形机器人STM32项目编译配置:"
	@echo "BINARY:     $(BINARY)"
	@echo "SRCFILES:   $(SRCFILES)"
	@echo "LDSCRIPT:   $(LDSCRIPT)"
	@echo "MCU目标:    STM32F407VET6"
	@echo "移植层:     $(FREERTOS_PORT)"
	@echo "FreeRTOS:   $(FREERTOS_KERNEL)"
	@echo "工具链:     $(PREFIX)"

.PHONY: monitor screen f407 f407-program program info

# End Makefile