## Pytorch cifar10练习


##### 总结

* 采用csdn上改进了的网络, 优化器, 比pytorch官网样例准确率要高
* 添加了一些命令行参数


##### 参考资料

* https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
* https://blog.csdn.net/shi2xian2wei2/article/details/84308644



##### 相关配置信息
``` 
----------Python Info----------
Version      : 3.7.3
Compiler     : GCC 8.3.0
Build        : ('default', 'Apr  3 2019 05:39:12')
Arch         : ('64bit', 'ELF')
------------Pip Info-----------
Version      : 18.1
Directory    : /usr/lib/python3/dist-packages/pip
----------System Info----------
Platform     : Linux-4.19.0-6-amd64-x86_64-with-debian-10.2
system       : Linux
node         : ian-PC
release      : 4.19.0-6-amd64
version      : #1 SMP Debian 4.19.67-2+deb10u2 (2019-11-11)
----------Hardware Info----------
machine      : x86_64
processor    : 
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
Address sizes:       39 bits physical, 48 bits virtual
CPU(s):              12
On-line CPU(s) list: 0-11
Thread(s) per core:  2
Core(s) per socket:  6
Socket(s):           1
NUMA node(s):        1
Vendor ID:           GenuineIntel
CPU family:          6
Model:               158
Model name:          Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz
Stepping:            10
CPU MHz:             3400.175
CPU max MHz:         4100.0000
CPU min MHz:         800.0000
BogoMIPS:            4416.00
Virtualization:      VT-x
L1d cache:           32K
L1i cache:           32K
L2 cache:            256K
L3 cache:            9216K
NUMA node0 CPU(s):   0-11
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_known_freq pni pclmulqdq dtes64 monitor ds_cpl vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb invpcid_single pti ssbd ibrs ibpb stibp tpr_shadow vnmi flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid mpx rdseed adx smap clflushopt intel_pt xsaveopt xsavec xgetbv1 xsaves dtherm ida arat pln pts hwp hwp_notify hwp_act_window hwp_epp flush_l1d
```