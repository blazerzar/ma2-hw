from sympy import Matrix, diff, exp, solve, sqrt, symbols

x, y = symbols('x y')
f = x**2 + exp(x) + y**2 - x * y

gradient = Matrix([diff(f, x), diff(f, y)])
norm_sq = gradient[0] ** 2 + gradient[1] ** 2
norm = sqrt(norm_sq)

norm_gradient = Matrix([diff(norm_sq, x), diff(norm_sq, y)])
norm_hessian = Matrix(
    [
        [diff(norm_gradient[0], x), diff(norm_gradient[0], y)],
        [diff(norm_gradient[1], x), diff(norm_gradient[1], y)],
    ]
)

# Solution inside [-2, 2] x [-2 ,2]
solution = solve(norm_gradient, (x, y))
x_1, y_1 = solution[1][0].evalf(), solution[1][1].evalf()
solution_hessian = norm_hessian.subs({x: x_1, y: y_1})

print(f'x₁ = {x_1:.5f},y₁ = {y_1:.5f}')
print('Norm hessian at (x₁, y₁):', solution_hessian)

# Corner points
for x_, y_ in [(-2, -2), (2, -2), (-2, 2), (2, 2)]:
    n = norm.subs({x: x_, y: y_}).evalf()
    print(f'||∇f({x_}, {y_})|| = {n:.5f}')

# Eigenvalues
lambda_1 = (4 + exp(x) + sqrt(4 + exp(2 * x))) / 2
lambda_2 = (4 + exp(x) - sqrt(4 + exp(2 * x))) / 2

for x_ in [-2, 2]:
    l1 = lambda_1.subs(x, x_).evalf()
    l2 = lambda_2.subs(x, x_).evalf()
    print(f'λ₁ = {l1:.5f}, λ₂ = {l2:.5f} at x = {x_}')
