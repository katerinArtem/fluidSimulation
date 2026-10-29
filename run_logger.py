"""
Логгер для прогонов решателя.
Имя файла: ns_{version}_{solver}_{case}_{nx}_att_{n}.txt
Пример:    ns_v1_s_k_64_att_1.txt
"""
import os
import datetime


class RunLogger:
    def __init__(self, base_dir, version, solver, case, nx, ny=None):
        self.base_dir = base_dir
        self.version = version
        self.solver  = solver
        self.case    = case
        self.nx      = nx
        self.ny      = ny if ny is not None else nx

        os.makedirs(base_dir, exist_ok=True)
        self.attempt = self._next_attempt()
        name = f"ns_{version}_{solver}_{case}_{nx}_att_{self.attempt}.txt"
        self.path = os.path.join(base_dir, name)
        self._f = open(self.path, "w", encoding="utf-8")
        self.t0 = datetime.datetime.now()
        self._write_header()

    def _next_attempt(self):
        pattern = f"ns_{self.version}_{self.solver}_{self.case}_{self.nx}_att_"
        n = 1
        while True:
            candidate = os.path.join(self.base_dir, f"{pattern}{n}.txt")
            if not os.path.exists(candidate):
                return n
            n += 1

    def _write_header(self):
        self._f.write(f"# Log: {self.path}\n")
        self._f.write(f"# Started: {self.t0.isoformat()}\n")
        self._f.write(f"# version={self.version} solver={self.solver} "
                      f"case={self.case} NX={self.nx} NY={self.ny} "
                      f"attempt={self.attempt}\n")
        self._f.write("#" + "-" * 68 + "\n")
        self._f.flush()

    def log(self, msg=""):
        print(msg)
        self._f.write(str(msg) + "\n")
        self._f.flush()

    def close(self):
        dt = (datetime.datetime.now() - self.t0).total_seconds()
        self._f.write(f"\n# Finished: {datetime.datetime.now().isoformat()}  "
                      f"(elapsed {dt:.1f} s)\n")
        self._f.close()


# --- Использование в scalar_paper.py ---
# from run_logger import RunLogger
#
# logger = RunLogger(
#     base_dir = "logs",
#     version  = "v1",
#     solver   = "s",          # 's' = scalar, 'v' = vectorized
#     case     = "k",          # 'k' = Kovasznay
#     nx       = NX,
#     ny       = NY,
# )
# logger.log(f"SIMPLE scalar | NX={NX} NY={NY} | RE={RE}")
# logger.log(f"hx={HX:.5f} hy={HY:.5f} DT={DT:.3e} LAMBDA={LAMBDA:.6f}")
# ... весь вывод через logger.log(...)
# logger.close()
#
# В конфиг стоит добавить флаг:
#   LOG_BASE_DIR = "logs"
#   LOG_VERSION  = "v1"