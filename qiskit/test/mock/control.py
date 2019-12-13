
"""
Fake backend supporting OpenPulse.
"""
from scipy import signal
from scipy.fftpack import fft,fftshift
import matplotlib.pyplot as plt 
import numpy as np
from qiskit.providers.models import (GateConfig, PulseBackendConfiguration,
                                     PulseDefaults, Command, UchannelLO)
from qiskit.qobj import PulseLibraryItem, PulseQobjInstruction 
from qiskit.pulse.commands import DelayInstruction
from .fake_backend import FakeBackend


class FakeNVCenters2Q(FakeBackend):
    """Trivial extension of the FakeOpenPulse2Q."""

    def __init__(self):
        configuration = PulseBackendConfiguration(
            backend_name='fake_nvcenters_2q',
            backend_version='0.0.0',
            n_qubits=2,
            meas_levels=[0, 1],
            basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
            simulator=False,
            local=True,
            conditional=True,
            open_pulse=True,
            memory=False,
            max_shots=65536,
            gates=[GateConfig(name='TODO', parameters=[], qasm_def='TODO')],
            coupling_map= [[0, 1]],
            n_registers=2,
            n_uchannels=2,
            u_channel_lo=[ [UchannelLO(q=0, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=-1. + 0.j), UchannelLO(q=1, scale=1. + 0.j)],
                [UchannelLO(q=0, scale=1. + 0.j)]
            ],
            meas_level=[0, 1],
            qubit_lo_range=[[0.01, 0.1], [0.01, 0.1]],
            meas_lo_range=[[100000, 800000], [100000, 800000]],
            dt=1,
            dtm=4e3,
            rep_times=[100, 250, 500, 1000],
            meas_map=[[0, 1]],
            channel_bandwidth=[
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3],
                [-0.02, 0.02], [-0.02, 0.02], [-0.02, 0.02],
                [-0.2, 0.4], [-0.3, 0.3], [-0.3, 0.3]
            ],
            discriminators=['threshold_counts'],
            acquisition_latency=[[100, 100], [100, 100], [100, 100]],
            conditional_latency=[
                [100, 1000], [1000, 100], [100, 1000],
                [100, 1000], [1000, 100], [100, 1000],
                [1000, 100], [100, 1000], [1000, 100]
            ]
        )

        base_sequence = [
            PulseLibraryItem(name='con', samples=[0]*2656),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2656*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2656),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2656),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2656*2),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2656),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2656),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2656*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2656)

        ]

        base_sequence2 = [
            PulseLibraryItem(name='uncon', samples=[0]*3186),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='uncon', samples=[0]*3186*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='uncon', samples=[0]*3186),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='uncon', samples=[0]*3186),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='uncon', samples=[0]*3186*2),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='uncon', samples=[0]*3186),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='uncon', samples=[0]*3186),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='uncon', samples=[0]*3186*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='uncon', samples=[0]*3186)

        ]
        z_sequence = [
            PulseLibraryItem(name='con', samples=[0]*2058),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2058*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2058),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2058),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2058*2),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2058),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2058),
            PulseQobjInstruction(name='x', ch='u0',t0=0, phase='0'),
            PulseLibraryItem(name='con', samples=[0]*2058*2),
            PulseQobjInstruction(name='y', ch='u0',t0=0, phase='np.pi/2'),
            PulseLibraryItem(name='con', samples=[0]*2058)

        ]

        self._defaults = PulseDefaults(
            qubit_freq_est=[0.03125, 0.03125],
            meas_freq_est=[563519657.894737, 563519657.894737],
            buffer=0,
            pulse_library=[PulseLibraryItem(name='test_pulse_1', samples=[np.exp, 0.1j]),
                           PulseLibraryItem(name='test_pulse_2', samples=[0.j, 0.1j, 1j]),
                           PulseLibraryItem(name='test_pulse_3', samples=[0.j, 0.1j, 1j, 0.5 + 0j]),
                           PulseLibraryItem(name='test_pulse_4',
                                            samples=7*[0.j, 0.1j, 1j, 0.5 + 0j])],
            cmd_def=[Command(name='u1', qubits=[0],
                             sequence=[PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u1', qubits=[1],
                             sequence=[PulseQobjInstruction(name='fc', ch='d1',t0=0, phase='-P1*np.pi')]),
                     
                     Command(name='u2', qubits=[0],
                             sequence=[PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P0*np.pi'),
                                       PulseQobjInstruction(name='test_pulse_4', ch='d0', t0=0),
                                       PulseQobjInstruction(name='fc', ch='d0',
                                                            t0=0, phase='-P1*np.pi')]),
                     Command(name='u2', qubits=[1],
                             sequence=[PulseQobjInstruction(name='fc', ch='d1',
                                                            t0=0, phase='-P0*np.pi'),  PulseQobjInstruction(name='test_pulse_4', ch='d1', t0=0),
                                       PulseQobjInstruction(name='fc', ch='d1',t0=0, phase='-P0*np.pi')]),
                     Command(name='u3', qubits=[0],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='d0', t0=0)]),
                     Command(name='u3', qubits=[1],
                             sequence=[PulseQobjInstruction(name='test_pulse_3', ch='d1', t0=0)]),
                     Command(name='cx', qubits=[0, 1],
                             sequence=base_sequence2*20 + base_sequence*16 + [PulseQobjInstruction(name='fc', ch='d0', t0=0, phase='np.pi/2')]),
                     Command(name='measure', qubits=[0],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='m0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m1', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m2', t0=0),
                                       PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                    memory_slot=[0])]),

                      Command(name='measure', qubits=[1],
                             sequence=[PulseQobjInstruction(name='test_pulse_1', ch='m0', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m1', t0=0),
                                       PulseQobjInstruction(name='test_pulse_1', ch='m2', t0=0),
                                       PulseQobjInstruction(name='acquire', duration=10, t0=0,
                                    memory_slot=[1])])]

        )

        super().__init__(configuration)

    def defaults(self):  # pylint: disable=missing-docstring
        return self._defaults