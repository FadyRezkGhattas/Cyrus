from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np

class FunctionalTask(ABC):
    def __init__(self, variables : Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def evaluate(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def get_init_x(self, key):
        raise NotImplementedError
    
class Ackley(FunctionalTask):
    def __init__(self, variables : Dict[str, Any]) -> None:
        """
        The Ackley function is widely used for testing optimization algorithms.
        In its two-dimensional form, as shown in the plot above, it is characterized
        by a nearly flat outer region, and a large hole at the centre. The function poses a risk
        for optimization algorithms, particularly hillclimbing algorithms, to be trapped in one of
        its many local minima. Doc Copyright: http://www.sfu.ca/~ssurjano/ackley.html

        Recommended variable values are: a = 20, b = 0.2 and c = 2π. 

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768],
            for all i = 1, …, d, although it may also be restricted to a smaller domain.

        Args:
            variables (Dict[str, Any]): Should contain a, b and c constant as well as dim (dimensionality of the problem)
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        sum1 = jnp.square(x).sum()
        sum2 = jnp.cos(self.variables['c']*x).sum()
        term1 = -self.variables['a'] * jnp.exp(-self.variables['b']*jnp.sqrt(sum1/self.variables['dim']))
        term2 = -jnp.exp(sum2/self.variables['dim'])
        return term1 + term2 + self.variables['a'] + jnp.exp(1)
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-32.768, maxval=32.768, shape=[self.variables['dim']])

class Matyas(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Matyas function is a commonly used optimization test function.
        It is defined as f(x, y) = 0.26 * (x^2 + y^2) - 0.48 * x * y.

        Input Domain:
            The function is usually evaluated on the hypercube x, y ∈ [-10, 10].

        Args:
            variables (Dict[str, Any]): Should contain 'dim' (dimensionality of the problem).
                Note that Matyas function is defined for 2-dimensional inputs.
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        x = x.flatten()
        checkify.check(len(x) == 2, "Matyas function requires 2-dimensional input.")

        term1 = 0.26 * (x[0] ** 2 + x[1] ** 2)
        term2 = -0.48 * x[0] * x[1]

        return term1 + term2

    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-10, maxval=10, shape=[self.variables['dim']])    

class Booth(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Booth function is a commonly used optimization problem for testing algorithms.
        It has several local minima and a single global minimum.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-10, 10],
            for all i = 1, 2.

        Args:
            variables (Dict[str, Any]): Should be empty as this function has no variables.
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        x = x.flatten()
        checkify.check(len(x) == 2, "Matyas function requires 2-dimensional input.")

        term1 = jnp.square(x[0] + 2*x[1] - 7)
        term2 = jnp.square(2*x[0] + x[1] - 5)
        return term1 + term2
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-10, maxval=10, shape=[2])

#TODO: chatgpt implementaiton is wrong. fix the evaluate function
class Rosenbrock(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Rosenbrock function, also known as the Rosenbrock's valley or the Banana function,
        is a non-convex function used as a performance test problem for optimization algorithms.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5, 10],
            for all i = 1, ..., d. This can also be restricted to the hypercube xi ∈ [-2.048, 2.048].

        Args:
            variables (Dict[str, Any]): Should contain dim (dimensionality of the problem)
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        x1 = x[:-1]
        x2 = x[1:]
        return jnp.sum(100.0 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-5, maxval=10, shape=[self.variables['dim']])

class Michalewicz(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Michalewicz function has d! local minima, and it is multimodal. The parameter m defines
        the steepness of they valleys and ridges; a larger m leads to a more difficult search. The
        recommended value of m is m = 10. The function's two-dimensional form is shown in the plot above.

        Global Minima:
        at d=2: f(x*)=-1.8013, at x*=(2.20,1.57)
        at d=5: f(x*)=-4.687658
        at d=10: f(x*)=-9.66015

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [0, π],
            for all i = 1, ..., d.

        Args:
            variables (Dict[str, Any]): Should contain dim (dimensionality of the problem)
        """
        super().__init__(variables)
        self.variables = variables
        self.indices = jnp.arange(1,self.variables['dim']+1)

    def evaluate(self, x):
        m = 10  # Parameter for the Michalewicz function
        term1 = jnp.sin(x)
        term2 = jnp.multiply(self.indices, x ** 2) / jnp.pi
        term2 = jnp.sin(term2)
        multiplication = term1 * (term2 ** (2 * m))
        return -jnp.sum(multiplication)
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=0, maxval=jnp.pi, shape=[self.variables['dim']])

class Beale(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Beale function is a multimodal, non-convex function used as a benchmark
        for testing optimization algorithms. It has several local minima and one global minimum.
        There are also sharp peaks at the corners of the input domain.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-4.5, 4.5],
            for all i = 1, 2.

        Args:
            variables (Dict[str, Any]): Unused in this function.
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        x = x.flatten()
        checkify.check(len(x) == 2, "Matyas function requires 2-dimensional input.")
        term1 = jnp.square(1.5 - x[0] + x[0] * x[1])
        term2 = jnp.square(2.25 - x[0] + x[0] * (x[1] ** 2))
        term3 = jnp.square(2.625 - x[0] + x[0] * (x[1] ** 3))
        return term1 + term2 + term3
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-4.5, maxval=4.5, shape=[2])

class Branin(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Branin function is a commonly used benchmark function for testing optimization algorithms.
        It is a multimodal function with three global minima and several local minima.
        The recommended values of a, b, c, r, s and t are:
        a = 1, b = 5.1/(4π2), c = 5/π, r = 6, s = 10 and t = 1/(8π). 

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5, 10],
            for all i = 1, 2.

        Args:
            variables (Dict[str, Any]): Should contain a, b, c, r, s and t.
        """
        super().__init__(variables)
        self.variables = variables
        self.a = self.variables['a']
        self.b = self.variables['b']
        self.c = self.variables['c']
        self.r = self.variables['r']
        self.s = self.variables['s']
        self.t = self.variables['t']


    def evaluate(self, x):
        term1 = self.a * (x[1] - self.b * x[0] ** 2 + self.c * x[0] - self.r) ** 2
        term2 = self.s * (1 - self.t) * jnp.cos(x[0])
        return term1 + term2 + self.s
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-5, maxval=10, shape=[2])

class StyblinskiTang(FunctionalTask):
    def __init__(self, variables: Dict[str, Any]) -> None:
        """
        The Styblinski-Tang function is a multimodal function used as a benchmark
        for testing optimization algorithms.

        Input Domain:
            The function is usually evaluated on the hypercube xi ∈ [-5, 5],
            for all i = 1, ..., d.

        Args:
            variables (Dict[str, Any]): Should contain dim (dimensionality of the problem)
        """
        super().__init__(variables)
        self.variables = variables

    def evaluate(self, x):
        term1 = 0.5 * jnp.sum((x**4) - 16 * (x**2) + 5 * x)
        return term1
    
    def get_init_x(self, key):
        return jax.random.uniform(key, minval=-5, maxval=5, shape=[self.variables['dim']])

if __name__ == '__main__':
    key = jax.random.PRNGKey(42)

    rosen = Rosenbrock({"dim": 4})
    f = rosen.evaluate(jnp.array([1.0, 1.0, 1.0, 1.0, 1.0]))

    mich = Michalewicz({"dim": 2})
    f = mich.evaluate(jnp.array([2.20, 1.57]))

    mich = StyblinskiTang({"dim": 2})
    f = mich.evaluate(jnp.array([-2.903534, -2.903534]))

    ackley = Ackley({"a": 20, "b": 0.2, "c": 2*jnp.pi, "dim": 30})
    params = ackley.get_init_x(key)
    f = ackley.evaluate(params)