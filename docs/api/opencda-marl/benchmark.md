# Benchmark Comparison API

!!! success "Implementation Status"
    The Benchmark Comparison System is **fully implemented** and provides comprehensive automated testing for comparing agent performance across different traffic scenarios.

The BenchmarkComparator provides a unified system for benchmarking different agent types (behavior, vanilla, rule_based) across standardized traffic scenarios (safe, balanced, aggressive) with comprehensive performance metrics.

```text
BenchmarkComparator
├── Configuration Management  # Dynamic YAML override system
├── Agent Testing            # Automated agent execution
├── Metrics Collection       # Performance analysis
├── Results Export          # JSON and visualization
└── Comparison Analysis     # Multi-agent comparison
```

## Core Classes

=== "BenchmarkComparator"

    Main class for automated benchmark comparison across agents and scenarios.

    ```python
    class BenchmarkComparator:
        """
        Unified benchmark comparison system for MARL agents.
        
        Features:
        - Dynamic configuration override for different traffic scenarios
        - Automated testing of multiple agent types
        - Comprehensive performance metrics collection
        - Results export and visualization
        """
    ```

=== "Constructor"

    ```python
    def __init__(self, results_dir: str = "benchmark_results"):
        """
        Initialize benchmark comparator.
        
        Parameters
        ----------
        results_dir : str
            Directory to store results and exported data
        """
    ```

### Configuration System

=== "Traffic Scenario Configurations"

    ```python
    # Built-in traffic scenario presets
    self.density_configs = {
        'safe': {
            'scenario_name': 'intersection',
            'rate_vph': 200,                    # Light traffic flow
            'expected_vehicles': 130,
            'expected_collision_rate': 5.0,     # ~5% collision rate
            'min_headway_s': 3.5,              # Conservative following
            'safety_time': 4.0,                # High safety threshold
            'emergency_param': 0.5,            # Moderate emergency braking
            'cautious_speed': 25.0,            # Conservative speed
            'spawn_num': 1,                    # Single vehicle per spawn
            'strategy': 'balanced'             # Fair distribution
        },
        'balanced': {
            'scenario_name': 'intersection', 
            'rate_vph': 400,                   # Moderate traffic flow
            'expected_vehicles': 270,
            'expected_collision_rate': 10.0,   # ~10% collision rate
            'min_headway_s': 2.0,             # Normal following distance
            'safety_time': 3.0,               # Standard safety threshold
            'emergency_param': 0.4,           # Balanced emergency braking
            'cautious_speed': 20.0,           # Normal speed
            'spawn_num': 2,                   # Multiple vehicles per spawn
            'strategy': 'balanced'
        },
        'aggressive': {
            'scenario_name': 'intersection',
            'rate_vph': 600,                  # Heavy traffic flow
            'expected_vehicles': 400,
            'expected_collision_rate': 20.0,  # ~20% collision rate
            'min_headway_s': 1.0,            # Tight following distance
            'safety_time': 2.0,              # Reduced safety threshold
            'emergency_param': 0.3,          # Minimal emergency braking
            'cautious_speed': 15.0,          # Reduced speed
            'spawn_num': 2,                  # Multiple vehicles per spawn
            'strategy': 'conflict'           # Conflict-maximizing distribution
        }
    }
    ```

### Key Methods

=== "Agent Testing"

    ```python
    def run_agent_test(
        self, 
        agent_type: str, 
        scenario: str, 
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run test for specific agent and scenario combination.
        
        Parameters
        ----------
        agent_type : str
            Agent type ('behavior', 'vanilla', 'rule_based')
        scenario : str
            Traffic scenario ('safe', 'balanced', 'aggressive')
        timeout : int
            Timeout in seconds (default: 300)
            
        Returns
        -------
        result : dict
            Test results with metrics:
            - agent: Agent type
            - scenario: Scenario name  
            - throughput_vpm: Vehicles per minute
            - success_rate: Percentage successful
            - collision_rate: Percentage collisions
            - total_vehicles: Total vehicles spawned
            - execution_time: Test duration
        """
    ```

=== "Configuration Override"

    ```python
    def create_modified_config(self, base_config_path: str, scenario: str) -> str:
        """
        Create modified configuration with scenario parameters.
        
        Parameters
        ----------
        base_config_path : str
            Path to base intersection.yaml
        scenario : str
            Scenario name for parameter override
            
        Returns
        -------
        temp_config_path : str
            Path to temporary modified configuration file
        """
        
        # Example override for 'aggressive' scenario:
        override_params = {
            'scenario.traffic_manager.rate_vph': 600,
            'scenario.traffic_manager.min_headway_s': 1.0,
            'agents.agent_behavior.safety_time': 2.0,
            'agents.agent_behavior.emergency_param': 0.3,
            'agents.rule_based.cautious_speed': 15.0
        }
    ```

=== "Metrics Parsing"

    ```python
    def parse_metrics_from_output(self, output: str) -> Dict[str, float]:
        """
        Parse performance metrics from coordinator output.
        
        Parameters
        ----------
        output : str
            Raw output from MARL coordinator
            
        Returns
        -------
        metrics : dict
            Parsed metrics:
            - throughput_vpm: Vehicles per minute
            - success_rate: Success percentage
            - collision_rate: Collision percentage  
            - total_vehicles: Total vehicle count
        """
        
        # Example parsing patterns:
        # "Throughput: 37.1 vehicles per minute"
        # "Success Rate: 83.3%"
        # "Collision Rate: 13.6%" 
        # "Total Vehicles Spawned: 270"
    ```

=== "Batch Testing"

    ```python
    def run_comprehensive_benchmark(
        self,
        agents: List[str] = None,
        scenarios: List[str] = None,
        timeout: int = 300
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run comprehensive benchmark across agents and scenarios.
        
        Parameters
        ----------
        agents : list, optional
            Agent types to test (default: all available)
        scenarios : list, optional  
            Scenarios to test (default: all available)
        timeout : int
            Timeout per test in seconds
            
        Returns
        -------
        results : dict
            Nested results (agent_scenario -> metrics)
        """
    ```

## Usage Examples

=== "Single Agent Test"

    ```python
    from test.marl.test_benchmark_comparison import BenchmarkComparator
    
    # Initialize comparator
    comparator = BenchmarkComparator()
    
    # Test single agent-scenario combination
    result = comparator.run_agent_test(
        agent_type='vanilla',
        scenario='balanced',
        timeout=300
    )
    
    print(f"Results for vanilla agent on balanced scenario:")
    print(f"  Throughput: {result['throughput_vpm']:.1f} vpm")
    print(f"  Success Rate: {result['success_rate']:.1f}%")
    print(f"  Collision Rate: {result['collision_rate']:.1f}%")
    ```

=== "Comprehensive Benchmark"

    ```python
    # Test all agents across all scenarios
    results = comparator.run_comprehensive_benchmark()
    
    # Analyze results
    for agent_scenario, metrics in results.items():
        print(f"{agent_scenario}:")
        print(f"  Throughput: {metrics['throughput_vpm']:.1f} vpm")
        print(f"  Success: {metrics['success_rate']:.1f}%")
        print(f"  Collisions: {metrics['collision_rate']:.1f}%")
    ```

=== "Specific Agent-Scenario Matrix"

    ```python
    # Test specific combinations
    results = comparator.run_comprehensive_benchmark(
        agents=['behavior', 'vanilla'],
        scenarios=['safe', 'aggressive'],
        timeout=180
    )
    
    # Results will contain:
    # 'safe_behavior', 'safe_vanilla', 'aggressive_behavior', 'aggressive_vanilla'
    ```

=== "Results Export"

    ```python
    # Export results to JSON
    results_file = comparator.export_results(results)
    print(f"Results exported to: {results_file}")
    
    # Generate comparison plots
    comparator.generate_comparison_plots(results)
    print("Comparison plots generated in benchmark_results/")
    ```

## CLI Integration

=== "Command Line Usage"

    ```bash
    # Test all agents with all scenarios
    python test/marl/test_benchmark_comparison.py --all-agents --all-scenarios
    
    # Test specific agents and scenarios
    python test/marl/test_benchmark_comparison.py --agents behavior vanilla --scenarios balanced
    
    # Quick test with custom timeout
    python test/marl/test_benchmark_comparison.py --agents rule_based --scenarios safe --timeout 60
    
    # Export results to specific directory
    python test/marl/test_benchmark_comparison.py --all-agents --all-scenarios --output-dir custom_results/
    ```

=== "Available Arguments"

    ```python
    parser.add_argument('--agents', nargs='+', 
                       choices=['behavior', 'vanilla', 'rule_based'],
                       help='Agent types to test')
    parser.add_argument('--scenarios', nargs='+',
                       choices=['safe', 'balanced', 'aggressive'], 
                       help='Traffic scenarios to test')
    parser.add_argument('--all-agents', action='store_true',
                       help='Test all available agents')
    parser.add_argument('--all-scenarios', action='store_true',
                       help='Test all available scenarios')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout per test in seconds')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    ```

## Performance Metrics

=== "Metric Definitions"

    | Metric | Description | Calculation | Unit |
    |--------|-------------|-------------|------|
    | **Throughput (VPM)** | Vehicles successfully crossing intersection | `completed_vehicles / (time_seconds / 60)` | vehicles/minute |
    | **Success Rate** | Percentage of vehicles reaching destination | `(completed / total) * 100` | percentage |
    | **Collision Rate** | Percentage of vehicles involved in collisions | `(collided / total) * 100` | percentage |
    | **Total Vehicles** | Actual number of vehicles spawned | Direct count from coordinator | count |

=== "Expected Performance Ranges"

    | Scenario | Expected Throughput | Expected Success | Expected Collisions |
    |----------|-------------------|------------------|-------------------|
    | **Safe** | 30-35 vpm | 85-90% | 3-7% |
    | **Balanced** | 35-40 vpm | 80-85% | 10-15% |
    | **Aggressive** | 40-45 vpm | 75-85% | 15-25% |

## Results Structure

=== "Individual Test Result"

    ```python
    {
        "agent": "vanilla",
        "scenario": "balanced", 
        "throughput_vpm": 37.1,
        "success_rate": 83.3,
        "collision_rate": 13.6,
        "total_vehicles": 270,
        "execution_time": 300.0,
        "timestamp": "2025-08-25T15:30:00",
        "config_used": "configs/marl/intersection_temp_balanced.yaml"
    }
    ```

=== "Comprehensive Results"

    ```python
    {
        "safe_behavior": {
            "throughput_vpm": 33.8,
            "success_rate": 86.1, 
            "collision_rate": 5.6,
            "total_vehicles": 167
        },
        "balanced_vanilla": {
            "throughput_vpm": 37.1,
            "success_rate": 83.3,
            "collision_rate": 13.6, 
            "total_vehicles": 270
        },
        # ... additional results
    }
    ```

## Integration Points

=== "Configuration System"

    ```python
    # Uses traffic scenario presets from configs/marl/default.yaml
    scenario_config = config.traffic_scenarios[scenario_name]
    
    # Creates temporary configuration files with overrides
    temp_config = create_modified_config(base_config, scenario_config)
    ```

=== "MARL Coordinator Integration"

    ```python
    # Executes tests through MARL coordinator
    cmd = f"timeout {timeout} pixi run python opencda.py -t intersection --marl --config {temp_config}"
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parses coordinator output for metrics
    metrics = parse_metrics_from_output(result.stdout)
    ```

=== "Results Export"

    ```python
    # Exports to JSON for further analysis
    results_file = f"benchmark_results/comparison_{timestamp}.json"
    
    # Generates visualization plots
    plot_file = f"benchmark_results/comparison_{timestamp}.png"
    ```

---

**Status**: ✅ **Fully Implemented** | **Location**: `test/marl/test_benchmark_comparison.py`