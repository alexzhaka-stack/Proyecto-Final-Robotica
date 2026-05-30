#!/usr/bin/env python3
from sympy import *
import matplotlib.pyplot as plt


class Robot():
  def __init__(self,
               l: tuple[float] = (0.3, 0.3),
               h: float = 0.3):

    # Variables articulares
    th1, th2, th3 = symbols("theta_1,theta_2,theta_3")

    # Longitudes del nuevo robot
    l1 = l[0]   # arm_link
    l2 = l[1]   # forearm_link

    # Cinemática directa del nuevo robot
    #
    # shoulder_joint:
    #   - gira sobre Z
    #   - actúa como torreta
    #
    # arm_joint:
    #   - está a una altura h = 0.3 m
    #   - gira sobre Y
    #
    # forearm_joint:
    #   - está al final del arm_link
    #   - gira sobre Y
    #
    T_0_1 = self.Rz(th1)
    T_1_2 = self.Tz(h) * self.Ry(th2)
    T_2_3 = self.Tx(l1) * self.Ry(th3)
    T_3_p = self.Tx(l2)

    T_0_p = T_0_1 * T_1_2 * T_2_3 * T_3_p
    T_0_p = simplify(T_0_p)

    # Vector de postura del efector final
    # Ahora el robot se mueve en 3D:
    # xi = [x, y, z]
    xi_0_p = Matrix([
      T_0_p[0, 3],
      T_0_p[1, 3],
      T_0_p[2, 3]
    ])

    # Jacobiano
    J = xi_0_p.jacobian([th1, th2, th3])
    J_inv = J.inv()

    # Velocidades del efector final
    x_dot, y_dot, z_dot = symbols("x_dot, y_dot, z_dot")

    # Construir polinomio lambda
    t = symbols("t")
    a_0, a_1, a_2, a_3, a_4, a_5 = symbols("a_0, a_1, a_2, a_3, a_4, a_5")

    lam = a_0 + a_1 * t + a_2 * t**2 + a_3 * t**3 + a_4 * t**4 + a_5 * t**5
    lam_dot = diff(lam, t)
    lam_dot_dot = diff(lam_dot, t)

    # Almacenar variables en el objeto
    self.th1, self.th2, self.th3 = th1, th2, th3

    self.T_0_p = T_0_p
    self.xi_0_p = xi_0_p
    self.J = J
    self.J_inv = J_inv

    self.x_dot = x_dot
    self.y_dot = y_dot
    self.z_dot = z_dot

    self.a_0 = a_0
    self.a_1 = a_1
    self.a_2 = a_2
    self.a_3 = a_3
    self.a_4 = a_4
    self.a_5 = a_5

    self.t = t
    self.lam = lam
    self.lam_dot = lam_dot
    self.lam_dot_dot = lam_dot_dot

  def def_tray(self,
               t_f: float = 2,
               frec: float = 15,
               th_i: tuple[float] = (0.0, -0.3, 0.6),
               xi_f: tuple[float] = (0.35, 0.20, 0.35)):

    # Posición inicial del efector final sustituyendo las juntas iniciales
    xi_i = self.xi_0_p.subs({
      self.th1: th_i[0],
      self.th2: th_i[1],
      self.th3: th_i[2]
    })

    # Muestreo y dt
    self.dt = 1.0 / frec
    self.muestras = int(t_f * frec) + 1

    # Ecuaciones de restricción para trayectoria
    eq1 = self.lam.subs({self.t: 0})
    eq2 = self.lam.subs({self.t: t_f}) - 1
    eq3 = self.lam_dot.subs({self.t: 0})
    eq4 = self.lam_dot.subs({self.t: t_f})
    eq5 = self.lam_dot_dot.subs({self.t: 0})
    eq6 = self.lam_dot_dot.subs({self.t: t_f})

    solutions = solve(
      (eq1, eq2, eq3, eq4, eq5, eq6),
      (self.a_0, self.a_1, self.a_2, self.a_3, self.a_4, self.a_5)
    )

    # Sustituyendo solución en polinomio lambda
    lam_s = self.lam.subs(solutions)
    lam_dot_s = self.lam_dot.subs(solutions)
    lam_dot_dot_s = self.lam_dot_dot.subs(solutions)

    # Ecuación de posiciones, velocidades y aceleraciones del efector final
    xi_f = Matrix([
      xi_f[0],
      xi_f[1],
      xi_f[2]
    ])

    xi_eq = xi_i + (xi_f - xi_i) * lam_s
    xi_dot_eq = (xi_f - xi_i) * lam_dot_s
    xi_dot_dot_eq = (xi_f - xi_i) * lam_dot_dot_s

    # Arreglo de tiempo
    t_m = Matrix.zeros(1, self.muestras)

    for i in range(self.muestras):
      t_m[i] = self.dt * i

    # Posición, velocidad y aceleración del efector final
    xi_m = Matrix.zeros(3, self.muestras)
    xi_dot_m = Matrix.zeros(3, self.muestras)
    xi_dot_dot_m = Matrix.zeros(3, self.muestras)

    # Muestreo del efector final
    for i in range(self.muestras):
      xi_m[:, i] = xi_eq.subs({self.t: t_m[i]})
      xi_dot_m[:, i] = xi_dot_eq.subs({self.t: t_m[i]})
      xi_dot_dot_m[:, i] = xi_dot_dot_eq.subs({self.t: t_m[i]})

    print("Posición final deseada del efector final:")
    print(xi_m[:, self.muestras - 1])

    # Cinemática inversa por velocidades
    th_dot_eq = self.J_inv * Matrix([
      self.x_dot,
      self.y_dot,
      self.z_dot
    ])

    # Posición, velocidad y aceleración de las juntas
    th_m = Matrix.zeros(3, self.muestras)
    th_dot_m = Matrix.zeros(3, self.muestras)
    th_dot_dot_m = Matrix.zeros(3, self.muestras)

    # Agregar valor inicial conocido
    th_m[:, 0] = Matrix([
      th_i[0],
      th_i[1],
      th_i[2]
    ])

    # Muestreo de las juntas
    for i in range(self.muestras):

      # Velocidades articulares
      th_dot_m[:, i] = th_dot_eq.subs({
        self.th1: th_m[0, i],
        self.th2: th_m[1, i],
        self.th3: th_m[2, i],
        self.x_dot: xi_dot_m[0, i],
        self.y_dot: xi_dot_m[1, i],
        self.z_dot: xi_dot_m[2, i]
      })

      th_dot_m[:, i] = th_dot_m[:, i].evalf()

      if i < self.muestras - 1:
        # Integración de Euler para obtener posiciones articulares
        th_m[:, i + 1] = th_m[:, i] + th_dot_m[:, i] * self.dt

      if i != 0:
        # Aceleración articular aproximada
        th_dot_dot_m[:, i - 1] = (th_dot_m[:, i] - th_dot_m[:, i - 1]) / self.dt

    # Guardar variables en la clase

    # Efector final
    self.xi_m = xi_m
    self.xi_dot_m = xi_dot_m
    self.xi_dot_dot_m = xi_dot_dot_m

    # Juntas
    self.th_m = th_m
    self.th_dot_m = th_dot_m
    self.th_dot_dot_m = th_dot_dot_m

    # Tiempo
    self.t_m = t_m

  def imp_tray(self):
    fig, (x_g, y_g, z_g) = plt.subplots(nrows=1, ncols=3)

    fig.suptitle("Posiciones del efector final")

    x_g.set_title("x")
    y_g.set_title("y")
    z_g.set_title("z")

    x_g.plot(self.t_m.T, self.xi_m[0, :].T, color="RED")
    y_g.plot(self.t_m.T, self.xi_m[1, :].T, color="green")
    z_g.plot(self.t_m.T, self.xi_m[2, :].T, color=(0, 0, 1))

    plt.show()

  def imp_junt(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows=1, ncols=3)

    fig.suptitle("Posiciones de las juntas")

    th1_g.set_title("shoulder_joint")
    th2_g.set_title("arm_joint")
    th3_g.set_title("forearm_joint")

    th1_g.plot(self.t_m.T, self.th_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T, self.th_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T, self.th_m[2, :].T, color=(0, 0, 1))

    plt.show()

  def imp_vel_junt(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows=1, ncols=3)

    fig.suptitle("Velocidades de las juntas")

    th1_g.set_title("shoulder_joint dot")
    th2_g.set_title("arm_joint dot")
    th3_g.set_title("forearm_joint dot")

    th1_g.plot(self.t_m.T, self.th_dot_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T, self.th_dot_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T, self.th_dot_m[2, :].T, color=(0, 0, 1))

    plt.show()

  def imp_acc_junt(self):
    fig, (th1_g, th2_g, th3_g) = plt.subplots(nrows=1, ncols=3)

    fig.suptitle("Aceleraciones de las juntas")

    th1_g.set_title("shoulder_joint ddot")
    th2_g.set_title("arm_joint ddot")
    th3_g.set_title("forearm_joint ddot")

    th1_g.plot(self.t_m.T, self.th_dot_dot_m[0, :].T, color="RED")
    th2_g.plot(self.t_m.T, self.th_dot_dot_m[1, :].T, color="green")
    th3_g.plot(self.t_m.T, self.th_dot_dot_m[2, :].T, color=(0, 0, 1))

    plt.show()

  # -----------------------------
  # Matrices homogéneas básicas
  # -----------------------------

  def Tx(self, x):
    return Matrix([
      [1, 0, 0, x],
      [0, 1, 0, 0],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ])

  def Ty(self, y):
    return Matrix([
      [1, 0, 0, 0],
      [0, 1, 0, y],
      [0, 0, 1, 0],
      [0, 0, 0, 1]
    ])

  def Tz(self, z):
    return Matrix([
      [1, 0, 0, 0],
      [0, 1, 0, 0],
      [0, 0, 1, z],
      [0, 0, 0, 1]
    ])

  def Rx(self, th):
    return Matrix([
      [1, 0,        0,         0],
      [0, cos(th), -sin(th),  0],
      [0, sin(th),  cos(th),  0],
      [0, 0,        0,        1]
    ])

  def Ry(self, th):
    return Matrix([
      [cos(th),  0, sin(th), 0],
      [0,        1, 0,       0],
      [-sin(th), 0, cos(th), 0],
      [0,        0, 0,       1]
    ])

  def Rz(self, th):
    return Matrix([
      [cos(th), -sin(th), 0, 0],
      [sin(th),  cos(th), 0, 0],
      [0,        0,       1, 0],
      [0,        0,       0, 1]
    ])

  def imprimir_cinematica(self):
    print("T_0_p:")
    pprint(self.T_0_p)

    print("\nxi_0_p:")
    pprint(self.xi_0_p)

    print("\nJ:")
    pprint(self.J)


def main():
  robot = Robot()

  robot.def_tray()

  robot.imp_tray()
  robot.imp_junt()


if __name__ == "__main__":
  main()