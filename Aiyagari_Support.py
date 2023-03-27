"""
Consumption-saving models with aggregate productivity shocks as well as idiosyn-
cratic income shocks.  Currently only contains one microeconomic model with a
basic solver.  Also includes a subclass of Market called CobbDouglas economy,
used for solving "macroeconomic" models with aggregate shocks.
"""
import numpy as np
import scipy.stats as stats
from HARK.interpolation import (
    LinearInterp,
    LinearInterpOnInterp1D,
    ConstantFunction,
    IdentityFunction,
    VariableLowerBoundFunc2D,
    BilinearInterp,
    LowerEnvelope2D,
    UpperEnvelope,
    MargValueFuncCRRA
)
from HARK.utilities import (
    CRRAutility,
    CRRAutilityP,
    CRRAutilityPP,
    CRRAutilityP_inv,
    CRRAutility_invP,
    CRRAutility_inv,
    make_grid_exp_mult
)
from HARK.distribution import (
    MarkovProcess,
    MeanOneLogNormal,
    Uniform,
    combine_indep_dstns,
    calc_expectation,
)
from HARK.ConsumptionSaving.ConsIndShockModel import (
    ConsumerSolution,
    IndShockConsumerType,
    init_idiosyncratic_shocks,
)
from HARK.ConsumptionSaving.ConsMarkovModel import MarkovConsumerType
from HARK import MetricObject, Market, AgentType
from copy import deepcopy
import matplotlib.pyplot as plt

__all__ = [
    "AggShockConsumerType",
    "AggShockMarkovConsumerType",
    "CobbDouglasEconomy",
    "SmallOpenEconomy",
    "CobbDouglasMarkovEconomy",
    "SmallOpenMarkovEconomy",
    "AggregateSavingRule",
    "AggShocksDynamicRule",
    "init_agg_shocks",
    "init_agg_mrkv_shocks",
    "init_cobb_douglas",
    "init_mrkv_cobb_douglas",
]

utility = CRRAutility
utilityP = CRRAutilityP
utilityPP = CRRAutilityPP
utilityP_inv = CRRAutilityP_inv
utility_invP = CRRAutility_invP
utility_inv = CRRAutility_inv




class MargValueFunc2D(MetricObject):
    """
    A class for representing a marginal value function in models where the
    standard envelope condition of dvdm(m,M) = u'(c(m,M)) holds (with CRRA utility).
    """

    distance_criteria = ["cFunc", "CRRA"]

    def __init__(self, cFunc, CRRA):
        """
        Constructor for a new marginal value function object.

        Parameters
        ----------
        cFunc : function
            A real function representing the marginal value function composed
            with the inverse marginal utility function, defined on normalized individual market
            resources and aggregate market resources-to-labor ratio: uP_inv(vPfunc(m,M)).
            Called cFunc because when standard envelope condition applies,
            uP_inv(vPfunc(m,M)) = cFunc(m,M).
        CRRA : float
            Coefficient of relative risk aversion.

        Returns
        -------
        new instance of MargValueFunc
        """
        self.cFunc = deepcopy(cFunc)
        self.CRRA = CRRA

    def __call__(self, m, M):
        return utilityP(self.cFunc(m, M), gam=self.CRRA)




###############################################################################

# Make a dictionary to specify an aggregate shocks consumer
init_agg_shocks = init_idiosyncratic_shocks.copy()
del init_agg_shocks["Rfree"]  # Interest factor is endogenous in agg shocks model
del init_agg_shocks["CubicBool"]  # Not supported yet for agg shocks model
del init_agg_shocks["vFuncBool"]  # Not supported yet for agg shocks model
init_agg_shocks["PermGroFac"] = [1.0]
# Grid of capital-to-labor-ratios (factors)
MgridBase = np.array(
    [0.1, 0.3, 0.6, 0.8, 0.9, 0.98, 1.0, 1.02, 1.1, 1.2, 1.6, 2.0, 3.0]
)
init_agg_shocks["MgridBase"] = MgridBase
init_agg_shocks["aXtraCount"] = 24
init_agg_shocks["aNrmInitStd"] = 0.0
init_agg_shocks["LivPrb"] = [0.98]



class AggShockConsumerType(IndShockConsumerType):
    """
    A class to represent consumers who face idiosyncratic (transitory and per-
    manent) shocks to their income and live in an economy that has aggregate
    (transitory and permanent) shocks to labor productivity.  As the capital-
    to-labor ratio varies in the economy, so does the wage rate and interest
    rate.  "Aggregate shock consumers" have beliefs about how the capital ratio
    evolves over time and take aggregate shocks into account when making their
    decision about how much to consume.
    """

    def __init__(self, **kwds):
        """
        Make a new instance of AggShockConsumerType, an extension of
        IndShockConsumerType.  Sets appropriate solver and input lists.
        """
        params = init_agg_shocks.copy()
        params.update(kwds)

        AgentType.__init__(
            self,
            solution_terminal=deepcopy(IndShockConsumerType.solution_terminal_),
            pseudo_terminal=False,
            **params
        )

        # Add consumer-type specific objects, copying to create independent versions
        self.time_vary = deepcopy(IndShockConsumerType.time_vary_)
        self.time_inv = deepcopy(IndShockConsumerType.time_inv_)
        self.del_from_time_inv("Rfree", "vFuncBool", "CubicBool")

        self.solve_one_period = solveConsAggShock
        self.update()

    def reset(self):
        """
        Initialize this type for a new simulated history of K/L ratio.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.initialize_sim()
        self.state_now['aLvlNow'] = self.kInit * np.ones(self.AgentCount)  # Start simulation near SS
        self.state_now['aNrm'] = self.state_now['aLvlNow'] / self.state_now['pLvl'] # ???

    def pre_solve(self):
        #        AgentType.pre_solve()
        self.update_solution_terminal()

    def update_solution_terminal(self):
        """
        Updates the terminal period solution for an aggregate shock consumer.
        Only fills in the consumption function and marginal value function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cFunc_terminal = BilinearInterp(
            np.array([[0.0, 0.0], [1.0, 1.0]]),
            np.array([0.0, 1.0]),
            np.array([0.0, 1.0]),
        )

        vPfunc_terminal = MargValueFuncCRRA(cFunc_terminal, self.CRRA)
        mNrmMin_terminal = ConstantFunction(0)
        self.solution_terminal = ConsumerSolution(
            cFunc=cFunc_terminal, vPfunc=vPfunc_terminal, mNrmMin=mNrmMin_terminal
        )

    def get_economy_data(self, economy):
        """
        Imports economy-determined objects into self from a Market.
        Instances of AggShockConsumerType "live" in some macroeconomy that has
        attributes relevant to their microeconomic model, like the relationship
        between the capital-to-labor ratio and the interest and wage rates; this
        method imports those attributes from an "economy" object and makes them
        attributes of the ConsumerType.

        Parameters
        ----------
        economy : Market
            The "macroeconomy" in which this instance "lives".  Might be of the
            subclass CobbDouglasEconomy, which has methods to generate the
            relevant attributes.

        Returns
        -------
        None
        """
        self.T_sim = (
            economy.act_T
        )  # Need to be able to track as many periods as economy runs
        self.kInit = economy.kSS  # Initialize simulation assets to steady state
        self.aNrmInitMean = np.log(
            0.00000001
        )  # Initialize newborn assets to nearly zero
        self.Mgrid = (
            economy.MSS * self.MgridBase
        )  # Aggregate market resources grid adjusted around SS capital ratio
        self.AFunc = economy.AFunc  # Next period's aggregate savings function
        self.Rfunc = economy.Rfunc  # Interest factor as function of capital ratio
        self.wFunc = economy.wFunc  # Wage rate as function of capital ratio
        self.DeprFac = economy.DeprFac  # Rate of capital depreciation
        self.PermGroFacAgg = (
            economy.PermGroFacAgg
        )  # Aggregate permanent productivity growth
        self.add_AggShkDstn(
            economy.AggShkDstn
        )  # Combine idiosyncratic and aggregate shocks into one dstn
        self.add_to_time_inv(
            "Mgrid", "AFunc", "Rfunc", "wFunc", "DeprFac", "PermGroFacAgg"
        )

    def add_AggShkDstn(self, AggShkDstn):
        """
        Updates attribute IncShkDstn by combining idiosyncratic shocks with aggregate shocks.

        Parameters
        ----------
        AggShkDstn : [np.array]
            Aggregate productivity shock distribution.  First element is proba-
            bilities, second element is agg permanent shocks, third element is
            agg transitory shocks.

        Returns
        -------
        None
        """
        if len(self.IncShkDstn[0].X) > 2:
            self.IncShkDstn = self.IncShkDstnWithoutAggShocks
        else:
            self.IncShkDstnWithoutAggShocks = self.IncShkDstn
        self.IncShkDstn = [
            combine_indep_dstns(self.IncShkDstn[t], AggShkDstn)
            for t in range(self.T_cycle)
        ]

    def sim_birth(self, which_agents):
        """
        Makes new consumers for the given indices.  Initialized variables include aNrm and pLvl, as
        well as time variables t_age and t_cycle.  Normalized assets and permanent income levels
        are drawn from lognormal distributions given by aNrmInitMean and aNrmInitStd (etc).

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        IndShockConsumerType.sim_birth(self, which_agents)
        if 'aLvl' in self.state_now and self.state_now['aLvl'] is not None:
            self.state_now['aLvl'][which_agents] = (
                self.state_now['aNrm'][which_agents] * self.state_now['pLvl'][which_agents]
            )
        else:
            self.state_now['aLvl'] = self.state_now['aNrm'] * self.state_now['pLvl']

    def sim_death(self):
        """
        Randomly determine which consumers die, and distribute their wealth among the survivors.
        This method only works if there is only one period in the cycle.

        Parameters
        ----------
        None

        Returns
        -------
        who_dies : np.array(bool)
            Boolean array of size AgentCount indicating which agents die.
        """
        # Divide agents into wealth groups, kill one random agent per wealth group
        #        order = np.argsort(self.aLvlNow)
        #        how_many_die = int(self.AgentCount*(1.0-self.LivPrb[0]))
        #        group_size = self.AgentCount/how_many_die # This should be an integer
        #        base_idx = self.RNG.randint(0,group_size,size=how_many_die)
        #        kill_by_rank = np.arange(how_many_die,dtype=int)*group_size + base_idx
        #        who_dies = np.zeros(self.AgentCount,dtype=bool)
        #        who_dies[order[kill_by_rank]] = True

        # Just select a random set of agents to die
        how_many_die = int(round(self.AgentCount * (1.0 - self.LivPrb[0])))
        base_bool = np.zeros(self.AgentCount, dtype=bool)
        base_bool[0:how_many_die] = True
        who_dies = self.RNG.permutation(base_bool)
        if self.T_age is not None:
            who_dies[self.t_age >= self.T_age] = True

        # Divide up the wealth of those who die, giving it to those who survive
        who_lives = np.logical_not(who_dies)
        wealth_living = np.sum(self.state_now['aLvl'][who_lives])
        wealth_dead = np.sum(self.state_now['aLvl'][who_dies])
        Ractuarial = 1.0 + wealth_dead / wealth_living
        self.state_now['aNrm'][who_lives] = self.state_now['aNrm'][who_lives] * Ractuarial
        self.state_now['aLvl'][who_lives] = self.state_now['aLvl'][who_lives] * Ractuarial
        return who_dies

    def get_Rfree(self):
        """
        Returns an array of size self.AgentCount with self.RfreeNow in every entry.

        Parameters
        ----------
        None

        Returns
        -------
        RfreeNow : np.array
             Array of size self.AgentCount with risk free interest rate for each agent.
        """
        RfreeNow = self.RfreeNow * np.ones(self.AgentCount)
        return RfreeNow

    def get_shocks(self):
        """
        Finds the effective permanent and transitory shocks this period by combining the aggregate
        and idiosyncratic shocks of each type.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        IndShockConsumerType.get_shocks(self)  # Update idiosyncratic shocks
        self.shocks['TranShk'] = (
            self.shocks['TranShk'] * self.TranShkAggNow * self.wRteNow
        )
        self.shocks['PermShk'] = self.shocks['PermShk'] * self.PermShkAggNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.get_MaggNow()
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            cNrmNow[these] = self.solution[t].cFunc(self.state_now['mNrm'][these], MaggNow[these])
            MPCnow[these] = self.solution[t].cFunc.derivativeX(
                self.state_now['mNrm'][these], MaggNow[these]
            )  # Marginal propensity to consume

        self.controls['cNrm'] = cNrmNow
        self.MPCnow = MPCnow
        return None

    def get_MaggNow(self):  # This function exists to be overwritten in StickyE model
        return self.MaggNow * np.ones(self.AgentCount)

    def market_action(self):
        """
        In the aggregate shocks model, the "market action" is to simulate one
        period of receiving income and choosing how much to consume.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.simulate(1)

    def calc_bounding_values(self):
        """
        Calculate human wealth plus minimum and maximum MPC in an infinite
        horizon model with only one period repeated indefinitely.  Store results
        as attributes of self.  Human wealth is the present discounted value of
        expected future income after receiving income this period, ignoring mort-
        ality.  The maximum MPC is the limit of the MPC as m --> mNrmMin.  The
        minimum MPC is the limit of the MPC as m --> infty.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        raise NotImplementedError()

    def make_euler_error_func(self, mMax=100, approx_inc_dstn=True):
        """
        Creates a "normalized Euler error" function for this instance, mapping
        from market resources to "consumption error per dollar of consumption."
        Stores result in attribute eulerErrorFunc as an interpolated function.
        Has option to use approximate income distribution stored in self.IncShkDstn
        or to use a (temporary) very dense approximation.

        NOT YET IMPLEMENTED FOR THIS CLASS

        Parameters
        ----------
        mMax : float
            Maximum normalized market resources for the Euler error function.
        approx_inc_dstn : Boolean
            Indicator for whether to use the approximate discrete income distri-
            bution stored in self.IncShkDstn[0], or to use a very accurate
            discrete approximation instead.  When True, uses approximation in
            IncShkDstn; when False, makes and uses a very dense approximation.

        Returns
        -------
        None

        Notes
        -----
        This method is not used by any other code in the library. Rather, it is here
        for expository and benchmarking purposes.

        """
        raise NotImplementedError()



# This example makes a high risk, low growth state and a low risk, high growth state
MrkvArray = np.array([[0.90, 0.10], [0.04, 0.96]])
PermShkAggStd = [
    0.012,
    0.006,
]  # Standard deviation of log aggregate permanent shocks by state
TranShkAggStd = [
    0.006,
    0.003,
]  # Standard deviation of log aggregate transitory shocks by state
PermGroFacAgg = [0.98, 1.02]  # Aggregate permanent income growth factor

# Make a dictionary to specify a Markov aggregate shocks consumer
init_agg_mrkv_shocks = init_agg_shocks.copy()
init_agg_mrkv_shocks["MrkvArray"] = MrkvArray


class AggShockMarkovConsumerType(AggShockConsumerType):
    """
    A class for representing ex ante heterogeneous "types" of consumers who
    experience both aggregate and idiosyncratic shocks to productivity (both
    permanent and transitory), who lives in an environment where the macroeconomic
    state is subject to Markov-style discrete state evolution.
    """

    def __init__(self, **kwds):
        params = init_agg_mrkv_shocks.copy()
        params.update(kwds)
        kwds = params
        AggShockConsumerType.__init__(self, **kwds)

        self.shocks['Mrkv'] = None

        self.add_to_time_inv("MrkvArray")
        self.solve_one_period = solve_ConsAggMarkov

    def add_AggShkDstn(self, AggShkDstn):
        """
        Variation on AggShockConsumerType.add_AggShkDstn that handles the Markov
        state. AggShkDstn is a list of aggregate productivity shock distributions
        for each Markov state.
        """
        if len(self.IncShkDstn[0][0].X) > 2:
            self.IncShkDstn = self.IncShkDstnWithoutAggShocks
        else:
            self.IncShkDstnWithoutAggShocks = self.IncShkDstn

        IncShkDstnOut = []
        N = self.MrkvArray.shape[0]
        for t in range(self.T_cycle):
            IncShkDstnOut.append(
                [
                    combine_indep_dstns(self.IncShkDstn[t][n], AggShkDstn[n])
                    for n in range(N)
                ]
            )
        self.IncShkDstn = IncShkDstnOut

    def update_solution_terminal(self):
        """
        Update the terminal period solution.  This method should be run when a
        new AgentType is created or when CRRA changes.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        AggShockConsumerType.update_solution_terminal(self)

        # Make replicated terminal period solution
        StateCount = self.MrkvArray.shape[0]
        self.solution_terminal.cFunc = StateCount * [self.solution_terminal.cFunc]
        self.solution_terminal.vPfunc = StateCount * [self.solution_terminal.vPfunc]
        self.solution_terminal.mNrmMin = StateCount * [self.solution_terminal.mNrmMin]

    def reset_rng(self):
        MarkovConsumerType.reset_rng(self)

    def get_shocks(self):
        """
        Gets permanent and transitory income shocks for this period.  Samples from IncShkDstn for
        each period in the cycle.  This is a copy-paste from IndShockConsumerType, with the
        addition of the Markov macroeconomic state.  Unfortunately, the get_shocks method for
        MarkovConsumerType cannot be used, as that method assumes that MrkvNow is a vector
        with a value for each agent, not just a single int.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        PermShkNow = np.zeros(self.AgentCount)  # Initialize shock arrays
        TranShkNow = np.zeros(self.AgentCount)
        newborn = self.t_age == 0
        for t in range(self.T_cycle):
            these = t == self.t_cycle
            N = np.sum(these)
            if N > 0:
                IncShkDstnNow = self.IncShkDstn[t - 1][
                    self.shocks['Mrkv']
                ]  # set current income distribution
                PermGroFacNow = self.PermGroFac[t - 1]  # and permanent growth factor

                # Get random draws of income shocks from the discrete distribution
                ShockDraws = IncShkDstnNow.draw(N, exact_match=True)
                # Permanent "shock" includes expected growth
                PermShkNow[these] = ShockDraws[0] * PermGroFacNow
                TranShkNow[these] = ShockDraws[1]

        # That procedure used the *last* period in the sequence for newborns, but that's not right
        # Redraw shocks for newborns, using the *first* period in the sequence.  Approximation.
        N = np.sum(newborn)
        if N > 0:
            these = newborn
            IncShkDstnNow = self.IncShkDstn[0][
                self.shocks['Mrkv']
            ]  # set current income distribution
            PermGroFacNow = self.PermGroFac[0]  # and permanent growth factor

            # Get random draws of income shocks from the discrete distribution
            ShockDraws = IncShkDstnNow.draw(N, exact_match=True)

            # Permanent "shock" includes expected growth
            PermShkNow[these] = ShockDraws[0] * PermGroFacNow
            TranShkNow[these] = ShockDraws[1]

        # Store the shocks in self
        self.EmpNow = np.ones(self.AgentCount, dtype=bool)
        self.EmpNow[TranShkNow == self.IncUnemp] = False
        self.shocks['TranShk'] = TranShkNow * self.TranShkAggNow * self.wRteNow
        self.shocks['PermShk'] = PermShkNow * self.PermShkAggNow

    def get_controls(self):
        """
        Calculates consumption for each consumer of this type using the consumption functions.
        For this AgentType class, MrkvNow is the same for all consumers.  However, in an
        extension with "macroeconomic inattention", consumers might misperceive the state
        and thus act as if they are in different states.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        cNrmNow = np.zeros(self.AgentCount) + np.nan
        MPCnow = np.zeros(self.AgentCount) + np.nan
        MaggNow = self.get_MaggNow()
        MrkvNow = self.getMrkvNow()

        StateCount = self.MrkvArray.shape[0]
        MrkvBoolArray = np.zeros((StateCount, self.AgentCount), dtype=bool)
        for i in range(StateCount):
            MrkvBoolArray[i, :] = i == MrkvNow

        for t in range(self.T_cycle):
            these = t == self.t_cycle
            for i in range(StateCount):
                those = np.logical_and(these, MrkvBoolArray[i, :])
                cNrmNow[those] = self.solution[t].cFunc[i](
                    self.state_now['mNrm'][those], MaggNow[those]
                )
                # Marginal propensity to consume
                MPCnow[those] = (
                    self.solution[t]
                    .cFunc[i]
                    .derivativeX(self.state_now['mNrm'][those], MaggNow[those])
                )
        self.controls['cNrm'] = cNrmNow
        self.MPCnow = MPCnow
        return None

    def getMrkvNow(self):  # This function exists to be overwritten in StickyE model
        return self.shocks['Mrkv'] * np.ones(self.AgentCount, dtype=int)

































































































###############################################################################

init_Aiyagari_agents = dict(LaborStatesNo=7, LaborAR=0.6, LaborSD=0.2, T_cycle=1, DiscFac=0.96, CRRA=1.0, LbrInd=1.0,
                            aMin=0.001, aMax=50.0, aCount=32, aNestFac=2, MgridBase=np.array(
        [0.1, 0.3, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0, 1.02, 1.05, 1.1, 1.2, 1.6, 2.0, 3.0]
    ), AgentCount=140)

#AgentCount has to be a multiple of the number of labor supply states "LaborStatesNo"!!

class AiyagariType(AgentType):
    """
    A class for representing agents in the Aiyagari (1994) model from
    the paper "Uninsured Idiosyncratic Risk and Aggregate Savings".  This class is derived form the Krusell-SmithType,
    but with generalized number of idiosyncratic levels of employment instead of just the two states of being employed or not.

    """

    def __init__(self, **kwds):
        """
        Make a new instance of the Aiyagari type.
        """
        params = init_Aiyagari_agents.copy()
        params.update(kwds)

        AgentType.__init__(self, pseudo_terminal=False, **params)

        # Add consumer-type specific objects
        self.time_vary = []
        self.time_inv = [
            "DiscFac",
            "CRRA",
        ]
        # need better handling of this
        self.state_now = {
            "aNow" : None,
            "mNow" : None,
            "EmpNow" : None,
            "LaborSupplyState": None
        }
        self.state_prev = {
            "aNow" : None,
            "mNow" : None,
            "EmpNow" : None,
            "LaborSupplyState": None
        }

        self.shock_vars = {
            "Mrkv" : None
        }

        self.solve_one_period = solve_Aiyagari



        self.update()

    def pre_solve(self):
        self.update()
        self.precompute_arrays()

    def update(self):
        """
        Construct objects used during solution from primitive parameters.
        """
        self.make_grid()
        self.update_solution_terminal()

    def get_economy_data(self, Economy):
        """
        Imports economy-determined objects into self from a Market.

        Parameters
        ----------
        Economy : AiyagariEconomy
            The "macroeconomy" in which this instance "lives".

        Returns
        -------
        None
        """
        self.T_sim = (
            Economy.act_T
        )  # Need to be able to track as many periods as economy runs
        self.kInit = Economy.KSS  # Initialize simulation assets to steady state
        self.MrkvInit = Economy.sow_init[
            "Mrkv"
        ]  # Starting Markov state for the macroeconomy
        self.Mgrid = (
            Economy.MSS * self.MgridBase
        )  # Aggregate market resources grid adjusted around SS capital ratio
        self.AFunc = Economy.AFunc  # Next period's aggregate savings function
        self.DeprFac = Economy.DeprFac  # Rate of capital depreciation
        self.CapShare = Economy.CapShare  # Capital's share of production
        self.LbrInd = Economy.LbrInd  # Idiosyncratic labor supply (when employed)
        self.UrateB = Economy.UrateB  # Unemployment rate in bad state
        self.UrateG = Economy.UrateG  # Unemployment rate in good state
        self.ProdB = Economy.ProdB  # Total factor productivity in bad state
        self.ProdG = Economy.ProdG  # Total factor productivity in good state
        self.MrkvIndArray = (
            Economy.MrkvIndArray
        )  # Transition probabilities among discrete states
        self.MrkvAggArray = (
            Economy.MrkvArray
        )  # Transition probabilities among aggregate discrete states
        self.MrkvEmplArray = (Economy.MrkvEmplArray) #Transition between employment and unemployment
        self.TauchenAux = (Economy.TauchenAux)
        self.add_to_time_inv(
            "Mgrid",
            "AFunc",
            "DeprFac",
            "CapShare",
            "LaborStatesNo",
            "LaborAR",
            "LaborSD",
            "UrateB",
            "LbrInd",
            "UrateG",
            "ProdB",
            "ProdG",
            "MrkvIndArray",
            "MrkvAggArray",
            "MrkvEmplArray",
            "TauchenAux",
        )

    def make_grid(self):
        """
        Construct the attribute aXtraGrid from the primitive attributes aMin,
        aMax, aCount, aNestFac.
        """
        self.aGrid = make_grid_exp_mult(self.aMin, self.aMax, self.aCount, self.aNestFac)
        self.add_to_time_inv("aGrid")

        import HARK

        SDshock = self.LaborSD * (1 - (self.LaborAR ** 2)) ** (0.5)

        self.TauchenAux = HARK.distribution.make_tauchen_ar1(self.LaborStatesNo, sigma=SDshock, ar_1=self.LaborAR,bound=3.0)
        # Approximating the autocorrelated labor supply equation with a seven state Markov process

        self.add_to_time_inv("TauchenAux")

    def update_solution_terminal(self):
        """
        Construct the trivial terminal period solution (initial guess).
        """


        cFunc_terminal = (4*self.LaborStatesNo) * [IdentityFunction(n_dims=2)]
        vPfunc_terminal = [
            MargValueFuncCRRA(cFunc_terminal[j], self.CRRA) for j in range(4*self.LaborStatesNo)
        ]
        self.solution_terminal = ConsumerSolution(
            cFunc=cFunc_terminal, vPfunc=vPfunc_terminal
        )

    def precompute_arrays(self):
        """
        Construct the attributes ProbArray, mNextArray, MnextArray, and RnextArray,
        which will be used by the one period solver.
        """
        # Get array sizes
        aCount = self.aGrid.size
        Mcount = self.Mgrid.size

        # Make tiled array of end-of-period idiosyncratic assets (order: a, M, s, s')
        aNow_tiled = np.tile(
            np.reshape(self.aGrid, [aCount, 1, 1, 1]), [1,Mcount, 4*self.LaborStatesNo, 4*self.LaborStatesNo]
        )



        # Make arrays of end-of-period aggregate assets (capital next period)
        AnowB = self.AFunc[0](self.Mgrid)
        AnowG = self.AFunc[1](self.Mgrid)
        KnextB = np.tile(np.reshape(AnowB, [1, Mcount, 1, 1]), [1, 1, 4*self.LaborStatesNo, 1])  #! delete: previously used [1, 1, 1, 4*self.LaborStatesNo]
        KnextG = np.tile(np.reshape(AnowG, [1, Mcount, 1, 1]), [1, 1, 4*self.LaborStatesNo, 1])  #! delete: previously used [1, 1, 1, 4*self.LaborStatesNo]
        Knext = np.concatenate((KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG, KnextB, KnextB, KnextG, KnextG), axis=3)    #! This needs adaptation if LaborStatesNo is changed

        #!N Knext above needs a full cycle of "KnextB, KnextB, KnextG, KnextG" per Aiyagari idiosyncratic labor supply state

        #!N Lnext: States are ordered as such: Bad-unemployed, Bad-empl, Good-unempl, Good-empl. repeated for each
        # Aiyagari LS state, thus per Aiyagari state we need two lines for the aggregate Labor supply. One for each of
        # the two aggregate states, leading to a labor supply of (1-Urate)*LbrInd, which is assumed to be fixed
        # Make arrays of aggregate labor and TFP next period
        Lnext = np.zeros((1, Mcount, 4*self.LaborStatesNo, 4*self.LaborStatesNo))
        Lnext[0, :, :, 0:2] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 2:4] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 4:6] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 6:8] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 8:10] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 10:12] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 12:14] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 14:16] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 16:18] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 18:20] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 20:22] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 22:24] = (1.0 - self.UrateG) * self.LbrInd
        Lnext[0, :, :, 24:26] = (1.0 - self.UrateB) * self.LbrInd
        Lnext[0, :, :, 26:28] = (1.0 - self.UrateG) * self.LbrInd


        #!N Similar as with Lnext; two lines per Aiyagari state, corresponding to the two aggregate states
        Znext = np.zeros((1, Mcount, 4*self.LaborStatesNo, 4*self.LaborStatesNo))
        Znext[0, :, :, 0:2] = self.ProdB
        Znext[0, :, :, 2:4] = self.ProdG
        Znext[0, :, :, 4:6] = self.ProdB
        Znext[0, :, :, 6:8] = self.ProdG
        Znext[0, :, :, 8:10] = self.ProdB
        Znext[0, :, :, 10:12] = self.ProdG
        Znext[0, :, :, 12:14] = self.ProdB
        Znext[0, :, :, 14:16] = self.ProdG
        Znext[0, :, :, 16:18] = self.ProdB
        Znext[0, :, :, 18:20] = self.ProdG
        Znext[0, :, :, 20:22] = self.ProdB
        Znext[0, :, :, 22:24] = self.ProdG
        Znext[0, :, :, 24:26] = self.ProdB
        Znext[0, :, :, 26:28] = self.ProdG

        # Calculate (net) interest factor and wage rate next period
        KtoLnext = Knext / Lnext
        Rnext = 1.0 + Znext * self.CapShare * KtoLnext ** (self.CapShare - 1.0) - self.DeprFac
        Wnext = Znext * (1.0 - self.CapShare) * KtoLnext ** self.CapShare

        # Calculate aggregate market resources next period
        Ynext = Znext * Knext ** self.CapShare * Lnext ** (1.0 - self.CapShare)
        Mnext = (1.0 - self.DeprFac) * Knext + Ynext

        # Tile the interest, wage, and aggregate market resources arrays
        Rnext_tiled = np.tile(Rnext, [aCount, 1, 1, 1])
        Wnext_tiled = np.tile(Wnext, [aCount, 1, 1, 1])
        Mnext_tiled = np.tile(Mnext, [aCount, 1, 1, 1])

        #######################################################################################

        LSStates = np.exp(self.TauchenAux[0])/np.mean(np.exp(self.TauchenAux[0]))

        #!N For each Aiyagari labor supply state there are four aggregate (KS) states: Bad-unempl, Bad-empl, Good-unempl, Good-empl
        #Thus one needs the repeating cycle of individual effective labor supply of: [0,LSStates[i],0,LSStates[i]]
        # Make an array of idiosyncratic effective labor supply next period; i.e. IdioLS*Indicator(employed)
        lNext_tiled = np.zeros([aCount, Mcount, 4*self.LaborStatesNo, 4*self.LaborStatesNo])
        lNext_tiled[:, :, :, 0] = LSStates[0]   #Bad-Unempl-State1 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 1] = LSStates[0]   #Bad-Empl-State1
        lNext_tiled[:, :, :, 2] = LSStates[0]   #Good-Unempl-State1 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 3] = LSStates[0]   #Good-Empl-State1
        lNext_tiled[:, :, :, 4] = LSStates[1]   #Bad-Unempl-State2 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 5] = LSStates[1]   #Bad-Empl-State2
        lNext_tiled[:, :, :, 6] = LSStates[1]   #Good-Unempl-State2 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 7] = LSStates[1]   #Good-Empl-State2
        lNext_tiled[:, :, :, 8] = LSStates[2]   #Bad-Unempl-State3 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 9] = LSStates[2]   #Bad-Empl-State3
        lNext_tiled[:, :, :, 10] = LSStates[2]  #Good-Unempl-State3 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 11] = LSStates[2]  #Good-Empl-State3
        lNext_tiled[:, :, :, 12] = LSStates[3]  #Bad-Unempl-State4 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 13] = LSStates[3]  #Bad-Empl-State4
        lNext_tiled[:, :, :, 14] = LSStates[3]  #Good-Unempl-State4 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 15] = LSStates[3]  #Good-Empl-State4
        lNext_tiled[:, :, :, 16] = LSStates[4]   #Bad-Unempl-State5 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 17] = LSStates[4]   #Bad-Empl-State5
        lNext_tiled[:, :, :, 18] = LSStates[4]   #Good-Unempl-State5 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 19] = LSStates[4]   #Good-Empl-State5
        lNext_tiled[:, :, :, 20] = LSStates[5]   #Bad-Unempl-State6 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 21] = LSStates[5]   #Bad-Empl-State6
        lNext_tiled[:, :, :, 22] = LSStates[5]   #Good-Unempl-State6 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 23] = LSStates[5]   #Good-Empl-State6
        lNext_tiled[:, :, :, 24] = LSStates[6]   #Bad-Unempl-State7 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 25] = LSStates[6]   #Bad-Empl-State7
        lNext_tiled[:, :, :, 26] = LSStates[6]   #Good-Unempl-State7 #! KS: this must be zero! Aiyagari: LSStates[0]
        lNext_tiled[:, :, :, 27] = LSStates[6]  #Good-Empl-State7




        # Calculate idiosyncratic market resources next period
        mNext = Rnext_tiled * aNow_tiled + Wnext_tiled * lNext_tiled

        # Make a tiled array of transition probabilities
        Probs_tiled = np.tile(
            np.reshape(self.MrkvIndArray, [1, 1, 4*self.LaborStatesNo, 4*self.LaborStatesNo]), [aCount, Mcount, 1, 1]
        )


        # Store the attributes that will be used by the solver
        self.ProbArray = Probs_tiled
        self.mNextArray = mNext
        self.MnextArray = Mnext_tiled
        self.RnextArray = Rnext_tiled
        self.add_to_time_inv("ProbArray", "mNextArray", "MnextArray", "RnextArray")




    def make_emp_idx_arrays(self):
        """
        Construct the attributes emp_permute and unemp_permute, each of which is
        a 2x2 nested list of boolean arrays.  The j,k-th element of emp_permute
        represents the employment states this period for agents who were employed
        last period when the macroeconomy is transitioning from state j to state k.
        Likewise, j,k-th element of unemp_permute represents the employment states
        this period for agents who were unemployed last period when the macro-
        economy is transitioning from state j to state k.  These attributes are
        referenced during simulation, when they are randomly permuted in order to
        maintain exact unemployment rates in each period.
        """
        # Get counts of employed and unemployed agents in each macroeconomic state
        B_unemp_N = int(np.round(self.UrateB * self.AgentCount))
        B_emp_N = self.AgentCount - B_unemp_N
        G_unemp_N = int(np.round(self.UrateG * self.AgentCount))
        G_emp_N = self.AgentCount - G_unemp_N


        # Bad-bad transition indices
        BB_stay_unemp_N = int(
            np.round(B_unemp_N * self.MrkvEmplArray[0, 0] / self.MrkvAggArray[0, 0])
        )
        BB_become_unemp_N = B_unemp_N - BB_stay_unemp_N
        BB_stay_emp_N = int(
            np.round(B_emp_N * (self.MrkvEmplArray[1, 1]) / self.MrkvAggArray[0, 0])
        )
        BB_become_emp_N = B_emp_N - BB_stay_emp_N
        BB_unemp_permute = np.concatenate(
            [
                np.ones(BB_become_emp_N, dtype=bool),
                np.zeros(BB_stay_unemp_N, dtype=bool),
            ]
        )
        BB_emp_permute = np.concatenate(
            [
                np.ones(BB_stay_emp_N, dtype=bool),
                np.zeros(BB_become_unemp_N, dtype=bool),
            ]
        )

        # Bad-good transition indices
        BG_stay_unemp_N = int(
            np.round(B_unemp_N * self.MrkvEmplArray[0, 2] / self.MrkvAggArray[0, 1])
        )
        BG_become_unemp_N = G_unemp_N - BG_stay_unemp_N
        BG_stay_emp_N = int(
            np.round(B_emp_N * (self.MrkvEmplArray[1, 3]) / self.MrkvAggArray[0, 1]))
        BG_become_emp_N = G_emp_N - BG_stay_emp_N
        BG_unemp_permute = np.concatenate(
            [
                np.ones(BG_become_emp_N, dtype=bool),
                np.zeros(BG_stay_unemp_N, dtype=bool),
            ]
        )
        BG_emp_permute = np.concatenate(
            [
                np.ones(BG_stay_emp_N, dtype=bool),
                np.zeros(BG_become_unemp_N, dtype=bool),
            ]
        )

        # Good-bad transition indices
        GB_stay_unemp_N = int(
            np.round(G_unemp_N * self.MrkvEmplArray[2, 0] / self.MrkvAggArray[1, 0])
        )
        GB_become_unemp_N = B_unemp_N - GB_stay_unemp_N
        GB_stay_emp_N = int(
            np.round(G_emp_N * self.MrkvEmplArray[3, 1] / self.MrkvAggArray[1, 0])
        )
        GB_become_emp_N = B_emp_N - GB_stay_emp_N
        GB_unemp_permute = np.concatenate(
            [
                np.ones(GB_become_emp_N, dtype=bool),
                np.zeros(GB_stay_unemp_N, dtype=bool),
            ]
        )
        GB_emp_permute = np.concatenate(
            [
                np.ones(GB_stay_emp_N, dtype=bool),
                np.zeros(GB_become_unemp_N, dtype=bool),
            ]
        )

        # Good-good transition indices
        GG_stay_unemp_N = int(
            np.round(G_unemp_N * self.MrkvEmplArray[2, 2] / self.MrkvAggArray[1, 1])
        )
        GG_become_unemp_N = G_unemp_N - GG_stay_unemp_N
        GG_stay_emp_N = int(
            np.round(G_emp_N * self.MrkvEmplArray[3, 3] / self.MrkvAggArray[1, 1])
        )
        GG_become_emp_N = G_emp_N - GG_stay_emp_N
        GG_unemp_permute = np.concatenate(
            [
                np.ones(GG_become_emp_N, dtype=bool),
                np.zeros(GG_stay_unemp_N, dtype=bool),
            ]
        )
        GG_emp_permute = np.concatenate(
            [
                np.ones(GG_stay_emp_N, dtype=bool),
                np.zeros(GG_become_unemp_N, dtype=bool),
            ]
        )

        # Store transition matrices as attributes of self
        self.unemp_permute = [
            [BB_unemp_permute, BG_unemp_permute],
            [GB_unemp_permute, GG_unemp_permute],
        ]
        self.emp_permute = [
            [BB_emp_permute, BG_emp_permute],
            [GB_emp_permute, GG_emp_permute],
        ]

    def reset(self):
        self.initialize_sim()

    def market_action(self):
        self.simulate(1)

    def initialize_sim(self):
        self.shocks['Mrkv'] = self.MrkvInit
        AgentType.initialize_sim(self)
        self.state_now["EmpNow"] = self.state_now["EmpNow"].astype(bool)
        self.state_now["LaborSupplyState"] = self.state_now["LaborSupplyState"]


        self.make_emp_idx_arrays()

    def sim_birth(self, which):
        """
        Create newborn agents with randomly drawn employment states.  This will
        only ever be called by initialize_sim() at the start of a new simulation
        history, as the Krusell-Smith/Aiyagari model does not have death and replacement.
        The sim_death() method does not exist, as AgentType's default of "no death"
        is the correct behavior for the model.
        """
        N = np.sum(which)
        if N == 0:
            return

        if self.shocks['Mrkv'] == 0:
            unemp_N = int(np.round(self.UrateB * N))
            emp_N = self.AgentCount - unemp_N
        elif self.shocks['Mrkv'] == 1:
            unemp_N = int(np.round(self.UrateG * N))
            emp_N = self.AgentCount - unemp_N
        else:
            assert False, "Illegal macroeconomic state: MrkvNow must be 0 or 1"
        EmpNew = np.concatenate(
            [np.zeros(unemp_N, dtype=bool), np.ones(emp_N, dtype=bool)]
        )



        #Introducing the new idiosyncratic labor supply states of Aiyagari 1994;
        # here I just initialize the agents' idiosyncratic labor supply states
        LSNew =np.empty(0)

        for n in range(self.LaborStatesNo):     #IMPORTANT: AgentCount/LaborStatesNo must be an integer number
            Aux = n*np.ones(int(self.AgentCount/self.LaborStatesNo))
            LSNew = np.concatenate((LSNew, Aux), axis = 0)

        LSNew2 =np.array([int(element) for element in LSNew])

#! TODO: Create if check to ensure AgentCount/LaborStatesNo is an integer


        self.state_now["EmpNow"][which] = self.RNG.permutation(EmpNew)
        self.state_now["aNow"][which] = self.kInit
        self.state_now["LaborSupplyState"][which] = self.RNG.permutation(LSNew2)


    def get_shocks(self):
        """
        Get new idiosyncratic employment states based on the macroeconomic state.
        """

        # Get boolean arrays for current employment states
        employed = self.state_prev["EmpNow"].copy().astype(bool)
        unemployed = np.logical_not(employed)

        # Indicator for Aggregate state of economy; derive from past unemployment rate rather than store previous value
        mrkv_prev = int((unemployed.sum() / float(self.AgentCount)) != self.UrateB)


        # Transition some agents between unemployment and employment
        emp_permute = self.emp_permute[mrkv_prev][self.shocks['Mrkv']]
        unemp_permute = self.unemp_permute[mrkv_prev][self.shocks['Mrkv']]

        EmpNow = self.state_now["EmpNow"]


        # It's really this permutation that delivers the shocks
        # This apparatus is trying to match the underlying Markov process.
        EmpNow[employed] = self.RNG.permutation(emp_permute)
        EmpNow[unemployed] = self.RNG.permutation(unemp_permute)



        #Introducing the shocks to idiosyncratic labor supply:
        transition_prob = self.TauchenAux[1]  # Markov transition matrix for the n states of labor supply

        LSStatePrev = [int(element) for element in self.state_prev["LaborSupplyState"]]
        LSStateNow = np.zeros(len(LSStatePrev))



        #Shocking agents' idiosyncratic labor supply given the tauchen approximation of the AR(1) process:
        for i in range(len(LSStatePrev)):
            LSStateNow[i] = np.random.choice(range(self.LaborStatesNo), size=None, replace=True, p=transition_prob[LSStatePrev[i]])

        self.state_now["LaborSupplyState"] = np.array(LSStateNow.copy())


    def get_states(self):
        """
        Get each agent's idiosyncratic state, their household market resources.
        """
        # !KS

        LSStates = np.exp(self.TauchenAux[0])/np.mean(np.exp(self.TauchenAux[0]))

        #!delete:
        # Approximating the autocorrelated labor supply equation with a seven state Markov process  #!KS
        #LSStates = self.TauchenAux[0] + self.LbrInd  # Discrete Markov States' deviation from aggregate labor supply, to be sensical we need to add 1 to all of them
        # In KS case one needs to get rid of labor supply differences; instead of the tauchen + self.LbrInd post "[self.LbrInd,self.LbrInd]"
        # and do not forget to set LbrInd in the Aiyagari Economy to the KS value!


        IdioLS = []

        for i in range(self.AgentCount):
            Aux = LSStates[int(self.state_now["LaborSupplyState"][i])]     #Translating the indicator of the LS state into the LS associated with that state
            IdioLS.append(Aux)


        #Calculating the vector of current market resources; the new step is the entry-wise multiplication of idiosyncratic
        # labor supply with the employment indicator, to get the effective labor supply that is remunerated for each agent
        self.state_now["mNow"] = self.Rnow * self.state_prev['aNow'] + self.Wnow * np.multiply(IdioLS,self.state_now["EmpNow"])


    def get_controls(self):
        """
        Get each agent's consumption given their current state.'
        """
        employed = self.state_now["EmpNow"].copy().astype(bool)
        unemployed = np.logical_not(employed)

        #!N: Adapt the number of lines to fit the number of Aiyagari states
        #Need boolean indicators for the idiosyncratic labor state of the agent:
        State1 = self.state_now["LaborSupplyState"] == 0
        State2 = self.state_now["LaborSupplyState"] == 1
        State3 = self.state_now["LaborSupplyState"] == 2
        State4 = self.state_now["LaborSupplyState"] == 3
        State5 = self.state_now["LaborSupplyState"] == 4
        State6 = self.state_now["LaborSupplyState"] == 5
        State7 = self.state_now["LaborSupplyState"] == 6


        #!N: Adapt the number of lines to fit the number of Aiyagari states
        #Indicators for different effective LS states:   (needs adaptation with LaborStatesNo neq 2)
        employed1 = employed * State1
        employed2 = employed * State2
        employed3 = employed * State3
        employed4 = employed * State4
        employed5 = employed * State5
        employed6 = employed * State6
        employed7 = employed * State7
        unemployed1 = unemployed * State1
        unemployed2 = unemployed * State2
        unemployed3 = unemployed * State3
        unemployed4 = unemployed * State4
        unemployed5 = unemployed * State5
        unemployed6 = unemployed * State6
        unemployed7 = unemployed * State7


        #!N: Mrkv = 0 means the Bad aggragate State, thus to keep with the order in the transition matrix between
        #the states, the index follows: 2 Bad states, 2 Good states, ... One full cycle per Aiyagari state
        # Get the discrete index for state of the agent; the state numbers come from the MrkvIndArray definition in the
        #Aiyagari Economy defined below
        if self.shocks['Mrkv'] == 0:  # Bad macroeconomic conditions
            unemp_1_idx = 0
            emp_1_idx = 1
            unemp_2_idx = 4
            emp_2_idx = 5
            unemp_3_idx = 8
            emp_3_idx = 9
            unemp_4_idx = 12
            emp_4_idx = 13
            unemp_5_idx = 16
            emp_5_idx = 17
            unemp_6_idx = 20
            emp_6_idx = 21
            unemp_7_idx = 24
            emp_7_idx = 25

        elif self.shocks['Mrkv'] == 1:  # Good macroeconomic conditions
            unemp_1_idx = 2
            emp_1_idx = 3
            unemp_2_idx = 6
            emp_2_idx = 7
            unemp_3_idx = 10
            emp_3_idx = 11
            unemp_4_idx = 14
            emp_4_idx = 15
            unemp_5_idx = 18
            emp_5_idx = 19
            unemp_6_idx = 22
            emp_6_idx = 23
            unemp_7_idx = 26
            emp_7_idx = 27
        else:
            assert False, "Illegal macroeconomic state: MrkvNow must be 0 or 1"



        #!N: Extend the list of using the appropriate cFunc to calculate the consumption of the employed or
        #unemployed agent for each Aiyagari state
        # Get consumption for each agent using the appropriate consumption function
        cNow = np.zeros(self.AgentCount)
        Mnow = self.Mnow * np.ones(self.AgentCount)
        cNow[unemployed1] = self.solution[0].cFunc[unemp_1_idx](
            self.state_now["mNow"][unemployed1], Mnow[unemployed1]
        )
        cNow[employed1] = self.solution[0].cFunc[emp_1_idx](
            self.state_now["mNow"][employed1], Mnow[employed1]
        )
        cNow[unemployed2] = self.solution[0].cFunc[unemp_2_idx](
            self.state_now["mNow"][unemployed2], Mnow[unemployed2]
        )
        cNow[employed2] = self.solution[0].cFunc[emp_2_idx](
            self.state_now["mNow"][employed2], Mnow[employed2]
        )
        cNow[unemployed3] = self.solution[0].cFunc[unemp_3_idx](
            self.state_now["mNow"][unemployed3], Mnow[unemployed3]
        )
        cNow[employed3] = self.solution[0].cFunc[emp_3_idx](
            self.state_now["mNow"][employed3], Mnow[employed3]
        )
        cNow[unemployed4] = self.solution[0].cFunc[unemp_4_idx](
            self.state_now["mNow"][unemployed4], Mnow[unemployed4]
        )
        cNow[employed4] = self.solution[0].cFunc[emp_4_idx](
            self.state_now["mNow"][employed4], Mnow[employed4]
        )
        cNow[unemployed5] = self.solution[0].cFunc[unemp_5_idx](
            self.state_now["mNow"][unemployed5], Mnow[unemployed5]
        )
        cNow[employed5] = self.solution[0].cFunc[emp_5_idx](
            self.state_now["mNow"][employed5], Mnow[employed5]
        )
        cNow[unemployed6] = self.solution[0].cFunc[unemp_6_idx](
            self.state_now["mNow"][unemployed6], Mnow[unemployed6]
        )
        cNow[employed6] = self.solution[0].cFunc[emp_6_idx](
            self.state_now["mNow"][employed6], Mnow[employed6]
        )
        cNow[unemployed7] = self.solution[0].cFunc[unemp_7_idx](
            self.state_now["mNow"][unemployed7], Mnow[unemployed7]
        )
        cNow[employed7] = self.solution[0].cFunc[emp_7_idx](
            self.state_now["mNow"][employed7], Mnow[employed7]
        )
        self.controls["cNow"] = cNow

    def get_poststates(self):
        """
        Gets each agent's retained assets after consumption.
        """
        self.state_now['aNow'] = self.state_now["mNow"] - self.controls["cNow"]





###############################################################################

def solve_Aiyagari(
        solution_next,
        DiscFac,
        CRRA,
        aGrid,
        Mgrid,
        mNextArray,
        MnextArray,
        ProbArray,
        RnextArray,
        LaborStatesNo,
):
    """
    Solve the one period problem of an agent in Aiyagari's 1994 model.
    Because this model is so specialized and only intended to be used with a very narrow
    case, many arrays can be precomputed, making the code here very short.  See the
    method KrusellSmithType.precompute_arrays() for details.

    Parameters
    ----------
    solution_next : ConsumerSolution
        Representation of the solution to next period's problem, including the
        discrete-state-conditional consumption function and marginal value function.
    DiscFac : float
        Intertemporal discount factor.
    CRRA : float
        Coefficient of relative risk aversion.
    aGrid : np.array
        Array of end-of-period asset values.
    Mgrid : np.array
        A grid of aggregate market resources in the economy.
    mNextArray : np.array
        Precomputed array of next period's market resources attained from every
        end-of-period state in the exogenous grid crossed with every shock that
        might attain.  Has shape [aCount, Mcount, 4, 4] ~ [a, M, s, s'].
    MnextArray : np.array
        Precomputed array of next period's aggregate market resources attained
        from every end-of-period state in the exogenous grid crossed with every
        shock that might attain.  Corresponds to mNextArray.
    ProbArray : np.array
        Tiled array of transition probabilities among discrete states.  Every
        slice [i,j,:,:] is identical and translated from MrkvIndArray.
    RnextArray : np.array
        Tiled array of net interest factors next period, attained from every
        end-of-period state crossed with every shock that might attain.

    Returns
    -------
    solution_now : ConsumerSolution
        Representation of this period's solution to the Krusell-Smith model.
    """
    #!N: n = number of Aiyagari states
    n = LaborStatesNo  #Number of idiosyncratic labor states

    # Loop over next period's state realizations, computing marginal value of market resources
    vPnext = np.zeros_like(mNextArray)
    for j in range(4*n):
        vPnext[:, :, :, j] = solution_next.vPfunc[j](
            mNextArray[:, :, :, j], MnextArray[:, :, :, j]
        )

    # Compute end-of-period marginal value of assets
    EndOfPrdvP = DiscFac * np.sum(RnextArray * vPnext * ProbArray, axis=3)



    # Invert the first order condition to find optimal consumption
    cNow = EndOfPrdvP ** (-1.0 / CRRA)




    # Find the endogenous gridpoints
    aCount = aGrid.size
    Mcount = Mgrid.size
    aNow = np.tile(np.reshape(aGrid, [aCount, 1, 1]), [1, Mcount, 4*n])
    mNow = aNow + cNow


    # Insert close to zero values at the bottom of both cNow and mNow arrays (consume nearly nothing if there a no market resources)
    cNow = np.concatenate([(np.zeros([1, Mcount, 4*n])+0.0000001), cNow], axis=0)
    mNow = np.concatenate([(np.zeros([1, Mcount, 4*n])+0.0000001), mNow], axis=0)



    # Construct the consumption and marginal value function for each discrete state
    cFunc_by_state = []
    vPfunc_by_state = []
    for j in range(4*n):
        cFunc_by_M = [LinearInterp(mNow[:, k, j], cNow[:, k, j]) for k in range(Mcount)]
        cFunc_j = LinearInterpOnInterp1D(cFunc_by_M, Mgrid)
        vPfunc_j = MargValueFuncCRRA(cFunc_j, CRRA)
        cFunc_by_state.append(cFunc_j)
        vPfunc_by_state.append(vPfunc_j)

    # Package and return the solution
    solution_now = ConsumerSolution(cFunc=cFunc_by_state, vPfunc=vPfunc_by_state)
    return solution_now


#######################################################################

init_Aiyagari_economy = {
    "verbose": True,
    "LaborStatesNo": 7,                                                                                             #!N
    "LaborAR": 0.6,
    "LaborSD": 0.2,    #Aiyagari: either 0.2 or 0.4
    "act_T": 11000,
    "T_discard": 1000,
    "DampingFac": 0.5,
    "intercept_prev": [0.0, 0.0],
    "slope_prev": [1.0, 1.0],
    "DiscFac": 0.96,     #KS value is 0.99     #Aiyagari: 0.96                                                      #!KS
    "CRRA": 1.0,
    "LbrInd": 1.0,  #Alan Lujan got 0.3271 indirectly from KS    #Aiyagari: 1                                       #!KS
    "ProdB": 1.0,     #KS: 0.99                      #Aiyagari:1                                                   #!KS
    "ProdG": 1.0,     # Original KS value is 1.01    #Aiyagari:1                                                   #!KS
    "CapShare": 0.36,
    "DeprFac": 0.08,  # Original KS value is 0.025 # Aiyagari 0.08                                                   #!KS
    "DurMeanB": 8.0,
    "DurMeanG": 8.0,
    "SpellMeanB": 2.5,  #KS:2.5
    "SpellMeanG": 1.5,  #KS:1.5
    "UrateB": 0.0,      #Original KS value 0.10     #Aiyagari: 0                                                    #!KS
    "UrateG": 0.0,      #Original KS value 0.04     #Aiyagari: 0                                                    #!KS
    "RelProbBG": 0.75,  #0.75
    "RelProbGB": 1.25,  #1.25
    "MrkvNow_init": 0,
}



class AiyagariEconomy(Market):
    """
    A class to represent an economy in the Aiyagari (1994) model.
    This model replicates the one presented in the JPE article "Income and Wealth
    Heterogeneity in the Macroeconomy", with its default parameters set to match
    those in the paper.

    Parameters
    ----------
    agents : [ConsumerType]
        List of types of consumers that live in this economy.
    tolerance: float
        Minimum acceptable distance between "dynamic rules" to consider the
        solution process converged.  Distance depends on intercept and slope
        of the log-linear "next capital ratio" function.
    act_T : int
        Number of periods to simulate when making a history of the market.
    """

    def __init__(self, agents=None, tolerance=0.01, **kwds):
        agents = agents if agents is not None else list()
        params = deepcopy(init_Aiyagari_economy)
        params.update(kwds)



        Market.__init__(
            self,
            agents=agents,
            tolerance=tolerance,
            sow_vars=["Mnow", "Aprev", "Mrkv", "Rnow", "Wnow"],
            reap_vars=["aNow", "EmpNow"],
            track_vars=["Mrkv", "Aprev", "Mnow", "Urate"],
            dyn_vars=["AFunc"],
            **params
        )
        self.update()

    def update(self):
        """
        Construct trivial initial guesses of the aggregate saving rules, as well
        as the perfect foresight steady state and associated objects.
        """


        StateCount = 2   #Macro states are still 2 (good and bad)
        AFunc_all = [
            AggregateSavingRule(self.intercept_prev[j], self.slope_prev[j])
            for j in range(StateCount)
        ]
        self.AFunc = AFunc_all
        self.KtoLSS = (
            (1.0 ** self.CRRA / self.DiscFac - (1.0 - self.DeprFac)) / self.CapShare
        ) ** (1.0 / (self.CapShare - 1.0))
        self.KSS = self.KtoLSS * self.LbrInd
        self.KtoYSS = self.KtoLSS ** (1.0 - self.CapShare)
        self.WSS = (1.0 - self.CapShare) * self.KtoLSS ** (self.CapShare)
        self.RSS = (
            1.0 + self.CapShare * self.KtoLSS ** (self.CapShare - 1.0) - self.DeprFac
        )
        self.MSS = self.KSS * self.RSS + self.WSS * self.LbrInd
        self.convertKtoY = lambda KtoY: KtoY ** (
            1.0 / (1.0 - self.CapShare)
        )  # converts K/Y to K/L
        self.rFunc = lambda k: self.CapShare * k ** (self.CapShare - 1.0)
        self.Wfunc = lambda k: ((1.0 - self.CapShare) * k ** (self.CapShare))
        self.sow_init["KtoLnow"] = self.KtoLSS
        self.sow_init["Mnow"] = self.MSS
        self.sow_init["Aprev"] = self.KSS
        self.sow_init["Rnow"] = self.RSS
        self.sow_init["Wnow"] = self.WSS
        self.PermShkAggNow_init = 1.0
        self.TranShkAggNow_init = 1.0
        self.sow_init["Mrkv"] = 0
        self.make_MrkvArray()

    def reset(self):
        """
        Reset the economy to prepare for a new simulation.  Sets the time index
        of aggregate shocks to zero and runs Market.reset().
        """
        self.Shk_idx = 0
        Market.reset(self)

    def make_MrkvArray(self):
        """
        Construct the attributes MrkvAggArray and MrkvIndArray from the primitive
        attributes DurMeanB, DurMeanG, SpellMeanB, SpellMeanG, UrateB, UrateG,
        RelProbGB, and RelProbBG.
        """

        # Construct aggregate Markov transition probabilities
        ProbBG = 1.0 / self.DurMeanB
        ProbGB = 1.0 / self.DurMeanG
        ProbBB = 1.0 - ProbBG
        ProbGG = 1.0 - ProbGB
        MrkvAggArray = np.array([[ProbBB, ProbBG], [ProbGB, ProbGG]])

        # Construct idiosyncratic Markov employment transition probabilities
        # ORDER: BU, BE, GU, GE
        MrkvEmplArray = np.zeros((4 , 4 ))

        # BAD-BAD QUADRANT
        MrkvEmplArray[0, 1] = ProbBB * 1.0 / self.SpellMeanB
        MrkvEmplArray[0, 0] = ProbBB * (1 - 1.0 / self.SpellMeanB)
        MrkvEmplArray[1, 0] = self.UrateB / (1.0 - self.UrateB) * MrkvEmplArray[0, 1]
        MrkvEmplArray[1, 1] = ProbBB - MrkvEmplArray[1, 0]

        # GOOD-GOOD QUADRANT
        MrkvEmplArray[2, 3] = ProbGG * 1.0 / self.SpellMeanG
        MrkvEmplArray[2, 2] = ProbGG * (1 - 1.0 / self.SpellMeanG)
        MrkvEmplArray[3, 2] = self.UrateG / (1.0 - self.UrateG) * MrkvEmplArray[2, 3]
        MrkvEmplArray[3, 3] = ProbGG - MrkvEmplArray[3, 2]

        # BAD-GOOD QUADRANT
        MrkvEmplArray[0, 2] = self.RelProbBG * MrkvEmplArray[2, 2] / ProbGG * ProbBG
        MrkvEmplArray[0, 3] = ProbBG - MrkvEmplArray[0, 2]
        MrkvEmplArray[1, 2] = (
                                     ProbBG * self.UrateG - self.UrateB * MrkvEmplArray[0, 2]
                             ) / (1.0 - self.UrateB)
        MrkvEmplArray[1, 3] = ProbBG - MrkvEmplArray[1, 2]

        # GOOD-BAD QUADRANT
        MrkvEmplArray[2, 0] = self.RelProbGB * MrkvEmplArray[0, 0] / ProbBB * ProbGB
        MrkvEmplArray[2, 1] = ProbGB - MrkvEmplArray[2, 0]
        MrkvEmplArray[3, 0] = (
                                     ProbGB * self.UrateB - self.UrateG * MrkvEmplArray[2, 0]
                             ) / (1.0 - self.UrateG)
        MrkvEmplArray[3, 1] = ProbGB - MrkvEmplArray[3, 0]


#######################################################################################
        #After having defined the transition matrices for the aggregate economy state and the idiosyncratic state
        # of employment, we need to create the transition matrix for the idiosyncratic labor supply states.
        #To this end I employ the Tauchen method to approximate the AR(1) process given in Aiyagari 1994 (The same method
        #Aiyagari uses to approximate the AR(1) in his paper.

        import HARK

        SDshock = self.LaborSD * ((1 - (self.LaborAR ** 2)) ** (0.5))

        TauchenAux = HARK.distribution.make_tauchen_ar1(self.LaborStatesNo, sigma= SDshock, ar_1=self.LaborAR,bound=3.0)
        #The Tauchen method gives back the logarithm of the idiosyncratic labor supply state deviations. Thus, we need
        #to transform them into the true labor supply states. We need to scale it by the average of the states' LSs to
        #ensure that the expected LS in aggregation is one.



        # Approximating the autocorrelated labor supply equation with a seven state Markov process

        #!N:
        #As the autocorrelated idiosyncratic shock process is independent of the aggregate state, one can simply multiply the
        #Markov matrix given by the tauchen approximation by the respective probability of transitioning from one aggregate-
        #employment state (Bad-unempl, Bad-empl, Good-unempl, Good-empl) to the next, which were calculated above under
        #MrkvEmplArray. The structure of the full Markov transition matrix must be to blow up the idiosyncratic
        #transition matrix given by the Tauchen method to replace each element of the Tauchen matrix with the full
        # MrkvEmplArray times the element to be replaced. Thus the resulting (n*4,n*4) matrix is still a valid stochastic
        #matrix with all rows and columns adding up to one.


        AuxMatrix1 = [element * TauchenAux[1][0,0] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix2 = [element * TauchenAux[1][0,1] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix3 = [element * TauchenAux[1][0,2] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix4 = [element * TauchenAux[1][0,3] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix5 = [element * TauchenAux[1][0,4] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix6 = [element * TauchenAux[1][0,5] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix7 = [element * TauchenAux[1][0,6] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix8 = [element * TauchenAux[1][1,0] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix9 = [element * TauchenAux[1][1,1] for element in MrkvEmplArray]  #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix10 = [element * TauchenAux[1][1,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix11 = [element * TauchenAux[1][1,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix12 = [element * TauchenAux[1][1,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix13 = [element * TauchenAux[1][1,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix14 = [element * TauchenAux[1][1,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix15 = [element * TauchenAux[1][2,0] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix16 = [element * TauchenAux[1][2,1] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix17 = [element * TauchenAux[1][2,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix18 = [element * TauchenAux[1][2,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix19 = [element * TauchenAux[1][2,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix20 = [element * TauchenAux[1][2,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix21 = [element * TauchenAux[1][2,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix22 = [element * TauchenAux[1][3,0] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix23 = [element * TauchenAux[1][3,1] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix24 = [element * TauchenAux[1][3,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix25 = [element * TauchenAux[1][3,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix26 = [element * TauchenAux[1][3,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix27 = [element * TauchenAux[1][3,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix28 = [element * TauchenAux[1][3,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix29 = [element * TauchenAux[1][4,0] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix30 = [element * TauchenAux[1][4,1] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix31 = [element * TauchenAux[1][4,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix32 = [element * TauchenAux[1][4,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix33 = [element * TauchenAux[1][4,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix34 = [element * TauchenAux[1][4,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix35 = [element * TauchenAux[1][4,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix36 = [element * TauchenAux[1][5,0] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix37 = [element * TauchenAux[1][5,1] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix38 = [element * TauchenAux[1][5,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix39 = [element * TauchenAux[1][5,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix40 = [element * TauchenAux[1][5,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix41 = [element * TauchenAux[1][5,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix42 = [element * TauchenAux[1][5,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix43 = [element * TauchenAux[1][6,0] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix44 = [element * TauchenAux[1][6,1] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix45 = [element * TauchenAux[1][6,2] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix46 = [element * TauchenAux[1][6,3] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix47 = [element * TauchenAux[1][6,4] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix48 = [element * TauchenAux[1][6,5] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS
        AuxMatrix49 = [element * TauchenAux[1][6,6] for element in MrkvEmplArray] #(1/self.LaborStatesNo)*MrkvEmplArray                #!KS

        #The above code created the sub matrices for the full transition matrix; these have to be concatenated
        #to form the rows of the final matrix below, to be later concatenated again on the vertical axis to form the
        #MrkvIndArray (the overall transition matrix). #!N

        AuxMatrix1R = np.concatenate((AuxMatrix1,AuxMatrix2,AuxMatrix3,AuxMatrix4,AuxMatrix5,AuxMatrix6,AuxMatrix7), axis=1)
        AuxMatrix2R = np.concatenate((AuxMatrix8,AuxMatrix9,AuxMatrix10,AuxMatrix11,AuxMatrix12,AuxMatrix13,AuxMatrix14), axis=1)
        AuxMatrix3R = np.concatenate((AuxMatrix15,AuxMatrix16,AuxMatrix17,AuxMatrix18,AuxMatrix19,AuxMatrix20,AuxMatrix21), axis=1)
        AuxMatrix4R = np.concatenate((AuxMatrix22,AuxMatrix23,AuxMatrix24,AuxMatrix25,AuxMatrix26,AuxMatrix27,AuxMatrix28), axis=1)
        AuxMatrix5R = np.concatenate((AuxMatrix29,AuxMatrix30,AuxMatrix31,AuxMatrix32,AuxMatrix33,AuxMatrix34,AuxMatrix35), axis=1)
        AuxMatrix6R = np.concatenate((AuxMatrix36,AuxMatrix37,AuxMatrix38,AuxMatrix39,AuxMatrix40,AuxMatrix41,AuxMatrix42), axis=1)
        AuxMatrix7R = np.concatenate((AuxMatrix43,AuxMatrix44,AuxMatrix45,AuxMatrix46,AuxMatrix47,AuxMatrix48,AuxMatrix49), axis=1)

        #The next line constructs the overall Markov matrix giving the transition probabilities for all 4*n states
        # each single agent can experience. #!N

        MrkvIndArray = np.concatenate((AuxMatrix1R,AuxMatrix2R,AuxMatrix3R,AuxMatrix4R,AuxMatrix5R,AuxMatrix6R,AuxMatrix7R), axis=0)

        # Test for valid idiosyncratic transition probabilities
        assert np.all(
            MrkvIndArray >= 0.0
        ), "Invalid idiosyncratic transition probabilities!"

        #Pass on the transition matrices
        self.MrkvArray = MrkvAggArray
        self.MrkvIndArray = MrkvIndArray
        self.MrkvEmplArray = MrkvEmplArray
        self.TauchenAux = TauchenAux

    def make_Mrkv_history(self):
        """
        Makes a history of macroeconomic Markov states, stored in the attribute
        MrkvNow_hist.  This variable is binary (0 bad, 1 good) in the KS model.
        """
        # Initialize the Markov history and set up transitions
        self.MrkvNow_hist = np.zeros(self.act_T, dtype=int)
        MrkvNow = self.MrkvNow_init

        markov_process = MarkovProcess(self.MrkvArray, seed= 0)
        for s in range(self.act_T):  # Add act_T_orig more periods
            self.MrkvNow_hist[s] = MrkvNow
            MrkvNow = markov_process.draw(MrkvNow)


    def mill_rule(self, aNow, EmpNow):
        """
        Method to calculate the capital to labor ratio, interest factor, and
        wage rate based on each agent's current state.  Just calls calc_R_and_W().

        See documentation for calc_R_and_W for more information.

        Returns
        -------
        Mnow : float
            Aggregate market resources for this period.
        Aprev : float
            Aggregate savings for the prior period.
        MrkvNow : int
            Binary indicator for bad (0) or good (1) macroeconomic state.
        Rnow : float
            Interest factor on assets in the economy this period.
        Wnow : float
            Wage rate for labor in the economy this period.
        """

        return self.calc_R_and_W(aNow, EmpNow)

    def calc_dynamics(self, Mnow, Aprev):
        """
        Method to update perceptions of the aggregate saving rule in each
        macroeconomic state; just calls calc_AFunc.
        """
        return self.calc_AFunc(Mnow, Aprev)


    def calc_R_and_W(self, aNow, EmpNow):
        """
        Calculates the interest factor and wage rate this period using each agent's
        capital stock to get the aggregate capital ratio.

        Parameters
        ----------
        aNow : [np.array]
            Agents' current end-of-period assets.  Elements of the list correspond
            to types in the economy, entries within arrays to agents of that type.
        EmpNow [np.array]
            Agents' binary employment states.  Not actually used in computation of
            interest and wage rates, but stored in the history to verify that the
            idiosyncratic unemployment probabilities are behaving as expected.

        Returns
        -------
        Mnow : float
            Aggregate market resources for this period.
        Aprev : float
            Aggregate savings for the prior period.
        MrkvNow : int
            Binary indicator for bad (0) or good (1) macroeconomic state.
        Rnow : float
            Interest factor on assets in the economy this period.
        Wnow : float
            Wage rate for labor in the economy this period.
        """
        # Calculate aggregate savings
        Aprev = np.mean(np.array(aNow))  # End-of-period savings from last period
        # Calculate aggregate capital this period
        AggK = Aprev  # ...becomes capital today

        # Calculate unemployment rate
        Urate = 1.0 - np.mean(np.array(EmpNow))
        self.Urate = Urate  # This is the unemployment rate for the *prior* period

        # Get this period's TFP and labor supply; i.e. aggregate state of the economy
        MrkvNow = self.MrkvNow_hist[self.Shk_idx]
        if MrkvNow == 0:
            Prod = self.ProdB
            AggL = (1.0 - self.UrateB) * self.LbrInd
        elif MrkvNow == 1:
            Prod = self.ProdG
            AggL = (1.0 - self.UrateG) * self.LbrInd
        self.Shk_idx += 1

        # Calculate the interest factor and wage rate this period
        KtoLnow = AggK / AggL
        Rnow = 1.0 + Prod * self.rFunc(KtoLnow) - self.DeprFac
        Wnow = Prod * self.Wfunc(KtoLnow)
        Mnow = Rnow * AggK + Wnow * AggL
        self.KtoLnow = KtoLnow  # Need to store this as it is a sow variable

        # Returns a tuple of these values
        return Mnow, Aprev, MrkvNow, Rnow, Wnow

    def calc_AFunc(self, Mnow, Aprev):
        """
        Calculate a new aggregate savings rule based on the history of the
        aggregate savings and aggregate market resources from a simulation.
        Calculates an aggregate saving rule for each macroeconomic Markov state.

        Parameters
        ----------
        Mnow : [float]
            List of the history of the simulated aggregate market resources for an economy.
        Anow : [float]
            List of the history of the simulated aggregate savings for an economy.

        Returns
        -------
        (unnamed) : CapDynamicRule
            Object containing new saving rules for each Markov state.
        """
        verbose = self.verbose
        discard_periods = (
            self.T_discard
        )  # Throw out the first T periods to allow the simulation to approach the SS
        update_weight = (
            1.0 - self.DampingFac
        )  # Proportional weight to put on new function vs old function parameters
        total_periods = len(Mnow)

        # Trim the histories of M_t and A_t and convert them to logs
        logAagg = np.log(Aprev[discard_periods:total_periods])
        logMagg = np.log(Mnow[discard_periods - 1 : total_periods - 1])
        MrkvHist = self.MrkvNow_hist[discard_periods - 1 : total_periods - 1]

        # For each Markov state, regress A_t on M_t and update the saving rule
        AFunc_list = []
        rSq_list = []
        for i in range(self.MrkvArray.shape[0]):
            these = i == MrkvHist
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                logMagg[these], logAagg[these]
            )

            # Make a new aggregate savings rule by combining the new regression parameters
            # with the previous guess
            intercept = (
                update_weight * intercept
                + (1.0 - update_weight) * self.intercept_prev[i]
            )
            slope = update_weight * slope + (1.0 - update_weight) * self.slope_prev[i]
            AFunc_list.append(
                AggregateSavingRule(intercept, slope)
            )  # Make a new next-period capital function
            rSq_list.append(r_value ** 2)

            # Save the new values as "previous" values for the next iteration
            self.intercept_prev[i] = intercept
            self.slope_prev[i] = slope

        # Print the new parameters
        if verbose:
            print(
                "intercept="
                + str(self.intercept_prev)
                + ", slope="
                + str(self.slope_prev)
                + ", r-sq="
                + str(rSq_list)
            )

        return AggShocksDynamicRule(AFunc_list)


###############################################################################



#########################################################

class AggregateSavingRule(MetricObject):
    """
    A class to represent agent beliefs about aggregate saving at the end of this period (AaggNow) as
    a function of (normalized) aggregate market resources at the beginning of the period (MaggNow).

    Parameters
    ----------
    intercept : float
        Intercept of the log-linear capital evolution rule.
    slope : float
        Slope of the log-linear capital evolution rule.
    """

    def __init__(self, intercept, slope):
        self.intercept = intercept
        self.slope = slope
        self.distance_criteria = ["slope", "intercept"]

    def __call__(self, Mnow):
        """
        Evaluates aggregate savings as a function of the aggregate market resources this period.

        Parameters
        ----------
        Mnow : float
            Aggregate market resources this period.

        Returns
        -------
        Aagg : Expected aggregate savings this period.
        """
        Aagg = np.exp(self.intercept + self.slope * np.log(Mnow))
        return Aagg


class AggShocksDynamicRule(MetricObject):
    """
    Just a container class for passing the dynamic rule in the aggregate shocks model to agents.

    Parameters
    ----------
    AFunc : CapitalEvoRule
        Aggregate savings as a function of aggregate market resources.
    """

    def __init__(self, AFunc):
        self.AFunc = AFunc
        self.distance_criteria = ["AFunc"]


