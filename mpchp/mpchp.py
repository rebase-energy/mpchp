from pyomo.environ import ConcreteModel, AbstractModel
from pyomo.environ import Set,Param,Var,Objective,Constraint
from pyomo.environ import PositiveIntegers, NonNegativeReals, Reals
from pyomo.environ import SolverFactory, minimize
from pyomo.environ import value
from pyomo.core.base.param import SimpleParam
import numpy as np


def solve_model(model_instance, solver):
    if 'path' in solver:
        optimizer = SolverFactory(solver['name'], executable=solver['path'])
    else:
        optimizer = SolverFactory(solver['name'])

    optimizer.solve(model_instance, tee=True, keepfiles=False)

    return model_instance


def mpchp(model_data):

    model = ConcreteModel()

    ## SETS
    model.T = Set(dimen=1, ordered=True, initialize=model_data[None]['T']) # Periods

    ## PARAMETERS
    model.demand                        = Param(model.T, within=Reals, initialize=model_data[None]['demand'])
    model.generation                    = Param(model.T, initialize=model_data[None]['generation'])

    model.battery_soc_min               = Param(initialize=model_data[None]['battery_soc_min'])
    model.battery_capacity              = Param(initialize=model_data[None]['battery_capacity'])
    model.battery_charge_max            = Param(initialize=model_data[None]['battery_charge_max'])
    model.battery_discharge_max         = Param(initialize=model_data[None]['battery_discharge_max'])
    model.battery_efficiency_charge     = Param(initialize=model_data[None]['battery_efficiency_charge'])
    model.battery_efficiency_discharge  = Param(initialize=model_data[None]['battery_efficiency_discharge'])

    model.energy_price_buy              = Param(model.T, initialize=model_data[None]['energy_price_buy'])
    model.energy_price_sell             = Param(model.T, initialize=model_data[None]['energy_price_sell'])
    model.grid_fee_energy               = Param(model.T, initialize=model_data[None]['grid_fee_energy'])
    model.grid_fee_power                = Param(initialize=model_data[None]['grid_fee_power'])
    model.grid_overcharge_penalty       = Param(initialize=model_data[None]['grid_overcharge_penalty'])
    model.grid_power_contract           = Param(initialize=model_data[None]['grid_power_contract'])

    model.hp_capacity                   = Param(model.T, initialize=model_data[None]['hp_capacity'])
    model.hp_cop                        = Param(model.T, initialize=model_data[None]['hp_cop'])
    model.hp_ramp_up_rate               = Param(initialize=model_data[None]['hp_ramp_up_rate'])
    model.hp_ramp_down_rate             = Param(initialize=model_data[None]['hp_ramp_down_rate'])
    
    model.temperature                   = Param(model.T, within=Reals, initialize=model_data[None]['temperature'])
    model.room_temperature_min          = Param(initialize=model_data[None]['room_temperature_min'])
    model.room_temperature_max          = Param(initialize=model_data[None]['room_temperature_max'])
    
    model.heat_capacity                 = Param(initialize=model_data[None]['heat_capacity'])
    model.heat_loss                     = Param(initialize=model_data[None]['heat_loss'])

    model.dt                            = Param(initialize=model_data[None]['dt'])
    
    # Initial/Final conditions
    model.battery_soc_ini               = Param(initialize=model_data[None]['battery_soc_ini'])
    model.battery_soc_fin               = Param(initialize=model_data[None]['battery_soc_fin'])
    model.heat_generation_ini           = Param(initialize=model_data[None]['heat_generation_ini'])
    model.room_temperature_ini          = Param(initialize=model_data[None]['room_temperature_ini'])
    model.room_temperature_fin          = Param(initialize=model_data[None]['room_temperature_fin'])


    ## VARIABLE LIMITS
    def soc_limits(model, t):
        return (model.battery_soc_min*model.battery_capacity, model.battery_capacity)
    def charge_limits(model, t):
        return (0.0, model.battery_charge_max)
    def discharge_limits(model, t):
        return (0.0, model.battery_discharge_max)
    def heat_limits(model, t):
        return (0.0, model.hp_capacity[t])
    def temperature_limits(model, t):
        return (model.room_temperature_min, model.room_temperature_max)  


    ## VARIABLES
    model.COST_ENERGY       = Var(model.T)
    model.COST_GRID_ENERGY  = Var(model.T)
    model.COST_GRID_POWER   = Var()
    model.P_CONTR           = Var(within=NonNegativeReals)
    model.P_OVER            = Var(within=NonNegativeReals)
    model.P_BUY             = Var(model.T, within=NonNegativeReals)
    model.P_SELL            = Var(model.T, within=NonNegativeReals)
    model.B_SOC             = Var(model.T, within=NonNegativeReals, bounds=soc_limits)
    model.B_IN              = Var(model.T, within=NonNegativeReals, bounds=charge_limits)
    model.B_OUT             = Var(model.T, within=NonNegativeReals, bounds=discharge_limits)
    model.Q                 = Var(model.T, within=NonNegativeReals, bounds=heat_limits)
    model.P_HEAT            = Var(model.T, within=NonNegativeReals)
    model.T_ROOM            = Var(model.T, within=NonNegativeReals, bounds=temperature_limits)


    ## OBJECTIVE
    # Minimize cost
    def total_cost(model):
        return sum(model.COST_ENERGY[t] + model.COST_GRID_ENERGY[t] for t in model.T) + model.COST_GRID_POWER
    model.total_cost = Objective(rule=total_cost, sense=minimize)


    ## CONSTRAINTS
    # Energy cost
    def energy_cost(model, t):
        return model.COST_ENERGY[t] == model.energy_price_buy[t]*model.P_BUY[t]*model.dt - model.energy_price_sell[t]*model.P_SELL[t]*model.dt
    model.energy_cost = Constraint(model.T, rule=energy_cost)

    # Grid energy cost
    def grid_energy_cost(model, t):
        return model.COST_GRID_ENERGY[t] == model.grid_fee_energy[t]*model.P_BUY[t]*model.dt
    model.grid_energy_cost = Constraint(model.T, rule=grid_energy_cost)

    # Grid power cost
    def grid_power_cost(model):
        return model.COST_GRID_POWER == model.grid_fee_power*model.P_CONTR + model.grid_overcharge_penalty*model.P_OVER
    model.grid_power_cost = Constraint(rule=grid_power_cost)

    # Overcharge
    def overcharge_import(model, t):
        return model.P_OVER >= model.P_BUY[t] - model.P_CONTR
    model.overcharge_import = Constraint(model.T, rule=overcharge_import)

    # Power balance
    def energy_balance(model, t):
        return model.P_SELL[t] - model.P_BUY[t] ==  model.generation[t] + model.B_OUT[t] - model.B_IN[t] - model.demand[t] - model.P_HEAT[t]
    model.energy_balance = Constraint(model.T, rule=energy_balance)

    # Heat generation
    def heat_generation(model, t):
        return model.Q[t] ==  model.hp_cop[t]*model.P_HEAT[t]
    model.heat_generation = Constraint(model.T, rule=heat_generation)
    
    # Heat ramp up
    def heat_ramp_up(model, t):
        if t==model.T.at(1):
            return model.Q[t] - model.heat_generation_ini <= model.hp_ramp_up_rate*model.hp_capacity[t]
        else:
            return model.Q[t] - model.Q[model.T.prev(t)] <= model.hp_ramp_up_rate*model.hp_capacity[t]
    model.heat_ramp_up = Constraint(model.T, rule=heat_ramp_up)

    # Heat ramp down
    def heat_ramp_down(model, t):
        if t==model.T.at(1):
            return model.heat_generation_ini - model.Q[t] <= model.hp_ramp_down_rate*model.hp_capacity[t]
        else:
            return model.Q[model.T.prev(t)] - model.Q[t] <= model.hp_ramp_down_rate*model.hp_capacity[t]
    model.heat_ramp_down = Constraint(model.T, rule=heat_ramp_down)   
    
    # Temperature model
    def temperature_function(model, t):
        if t==model.T.last():
            return model.room_temperature_fin - model.T_ROOM[t] == 3600*model.dt/model.heat_capacity*(model.Q[t] - model.heat_loss*(model.T_ROOM[t]-model.temperature[t]))  
        else:
            return model.T_ROOM[model.T.next(t)] == model.T_ROOM[t] + 3600*model.dt/model.heat_capacity*(model.Q[t] - model.heat_loss*(model.T_ROOM[t]-model.temperature[t]))                          
    model.temperature_function = Constraint(model.T, rule=temperature_function)
    
    # Battery energy balance
    def battery_soc(model, t):
        if t==model.T.first():
            return model.B_SOC[t] - model.battery_soc_ini*model.battery_capacity == model.battery_efficiency_charge*model.B_IN[t]*model.dt  - (1/model.battery_efficiency_discharge)*model.B_OUT[t]*model.dt
        else:
            return model.B_SOC[t] - model.B_SOC[model.T.prev(t)] == model.battery_efficiency_charge*model.B_IN[t]*model.dt  - (1/model.battery_efficiency_discharge)*model.B_OUT[t]*model.dt
    model.battery_soc = Constraint(model.T, rule=battery_soc)

    # Fix battery soc in the last period
    if value(model.battery_soc_fin) > 0:
        model.B_SOC[model.T.last()].fix(model.battery_soc_fin*model.battery_capacity)
    
    # Fix power contract (if 0 then power contract level is optimized)
    if value(model.grid_power_contract) > 0:
        model.P_CONTR.fix(model.grid_power_contract)

    # Fix first period room temperature
    model.T_ROOM[model.T.first()].fix(model.room_temperature_ini)

    return model


def mpchp_input(data):

    periods = np.arange(1, len(data['generation'])+1)
    
    demand = dict(zip(periods,  data['demand']))
    generation = dict(zip(periods,  data['generation']))
    temperature = dict(zip(periods,  list(map(lambda x:x+273.15, data['temperature'])))) # °C --> K
    
    heat_capacity = data['heat_capacity']
    heat_loss = data['heat_loss']  

    battery_capacity = data['battery_capacity'] if 'battery_capacity' in data else 0.0
    battery_soc_min = data['battery_soc_min'] if 'battery_soc_min' in data else 0.0
    battery_charge_max = data['battery_charge_max'] if 'battery_charge_max' in data else 0.0
    battery_discharge_max = data['battery_discharge_max'] if 'battery_discharge_max' in data else 0.0
    battery_efficiency_charge = data['battery_efficiency_charge'] if 'battery_efficiency_charge' in data else 1.0
    battery_efficiency_discharge = data['battery_efficiency_discharge'] if 'battery_efficiency_discharge' in data else 1.0
    battery_soc_ini = data['battery_soc_ini'] if 'battery_soc_ini' in data else 0.0
    battery_soc_fin = data['battery_soc_fin'] if 'battery_soc_fin' in data else 0.0 

    energy_price_buy = dict(zip(periods,  data['energy_price_buy']))
    energy_price_sell = dict(zip(periods,  data['energy_price_sell']))
    grid_fee_energy = dict(zip(periods,  data['grid_fee_energy']))
    grid_fee_power = data['grid_fee_power']
    grid_overcharge_penalty = data['grid_overcharge_penalty']
    grid_power_contract = data['grid_power_contract']

    hp_capacity = dict(zip(periods,  data['hp_capacity']))
    hp_cop = dict(zip(periods,  data['hp_cop']))
    hp_ramp_up_rate = data['hp_ramp_up_rate']
    hp_ramp_down_rate = data['hp_ramp_down_rate']

    room_temperature_min = data['room_temperature_min']+273.15
    room_temperature_max = data['room_temperature_max']+273.15
    room_temperature_ini = data['room_temperature_ini']+273.15
    room_temperature_fin = data['room_temperature_fin']+273.15
    
    heat_generation_ini = data['heat_generation_ini']

    dt = data['period_length'] if 'period_length' in data else 1 


    # Create model data input dictionary
    model_data = {None: {
        'T': periods,
        'demand': demand,
        'generation': generation,
        'temperature': temperature,
        'heat_capacity': heat_capacity,
        'heat_loss': heat_loss,
        'battery_capacity': battery_capacity,
        'battery_soc_min': battery_soc_min,
        'battery_charge_max': battery_charge_max,
        'battery_discharge_max': battery_discharge_max,
        'battery_efficiency_charge': battery_efficiency_charge,
        'battery_efficiency_discharge': battery_efficiency_discharge,
        'battery_soc_ini': battery_soc_ini,
        'battery_soc_fin': battery_soc_fin,
        'energy_price_buy': energy_price_buy,
        'energy_price_sell': energy_price_sell,
        'grid_fee_energy': grid_fee_energy,
        'grid_fee_power': grid_fee_power,
        'grid_overcharge_penalty': grid_overcharge_penalty,
        'grid_power_contract': grid_power_contract,
        'hp_capacity': hp_capacity,
        'hp_cop': hp_cop,
        'hp_ramp_up_rate': hp_ramp_up_rate,
        'hp_ramp_down_rate': hp_ramp_down_rate,
        'room_temperature_min': room_temperature_min,
        'room_temperature_max': room_temperature_max,
        'room_temperature_ini': room_temperature_ini,
        'room_temperature_fin': room_temperature_fin,
        'heat_generation_ini': heat_generation_ini,
        'dt': dt,
    }}

    return model_data


def mpchp_results(solution):
    
    s = dict()
    s['cost_energy'] = value(solution.COST_ENERGY[:])
    s['cost_grid_energy'] = value(solution.COST_GRID_ENERGY[:])
    s['cost_grid_power'] = value(solution.COST_GRID_POWER)
    
    s['power_overcharge'] = value(solution.P_OVER)
    s['power_buy'] = value(solution.P_BUY[:])
    s['power_sell'] = value(solution.P_SELL[:])
    s['battery_soc'] = value(solution.B_SOC[:])
    s['battery_charge'] = value(solution.B_IN[:])
    s['battery_discharge'] = value(solution.B_OUT[:])
    
    # In Pyomo 5.7.3 there is an inconsistency on how a VAR is called and how a VAR with fixed value
    if type(solution.P_CONTR.value) ==  SimpleParam:
        s['power_contract'] = value(solution.P_CONTR)()
    else:
        s['power_contract'] = value(solution.P_CONTR)
    
    s['heat_generation'] = value(solution.Q[:])  
    s['power_heatpump'] = value(solution.P_HEAT[:]) 
    s['room_temperature'] = list(map(lambda x:x-273.15, value(solution.T_ROOM[:])))  # K --> °C  
     
    return s