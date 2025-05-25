import pgmpy
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
model = BayesianNetwork([('Rain', 'WetGrass')])
cpd_rain = TabularCPD(variable='Rain', variable_card=2, values=[[0.7], [0.3]])
cpd_wetgrass = TabularCPD(
    variable='WetGrass',
    variable_card=2,
    values=[[0.9, 0.2],   # P(WetGrass=0 | Rain)
            [0.1, 0.8]],  # P(WetGrass=1 | Rain)
    evidence=['Rain'],
    evidence_card=[2]
)

model.add_cpds(cpd_rain, cpd_wetgrass)
print(model.check_model())
infer = VariableElimination(model)
result = infer.query(variables=['Rain'], evidence={'WetGrass': 1})
print(result)
