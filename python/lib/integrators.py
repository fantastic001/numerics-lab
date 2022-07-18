
class BaseIntegrator(object):
    
    def __init__(self):
        pass
    def make_step(self, x, v, t, dt, a):
        """
        Makes one step during integration 

        returns (x,v)
        """
        pass

    def integrate(self, x0, v0, a, t, dt):
        """
        Integrates over time t 

        returns (t, x, v) as list which are functions of time 
        """
        pass

class EulerIntegrator(BaseIntegrator):

    def make_step(self, x, v, t, dt, a):
        w = v + dt*a(x,v,t)
        s = x + w*dt
        return (s,w)

    def integrate(self, x0, v0, a, t, dt):
        x = [] 
        v = [] 
        steps = int(t/dt)
        if t/dt != steps:
            raise ValueError("Specify dt such that it gives integer number of steps")
        time = 0 
        time_axis = []
        x.append(x0)
        v.append(v0)
        for step in range(steps):
           time_axis.append(time)
           time += dt
           s,w = self.make_step(x[-1], v[-1], time, dt, a)
           x.append(s)
           v.append(w)
        time_axis.append(time)
        return (time_axis, x, v)


