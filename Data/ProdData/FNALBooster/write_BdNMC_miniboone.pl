#!/usr/bin/perl


# This perl function write an BdNMC parameter card


 if($#ARGV < 1 )
 {
 exit;
 }


# It also only consider production channels relevant


my $fmass = abs($ARGV[1])+abs($ARGV[2]);
$outname="parameter.dat";
$BdNMCout=$ARGV[3];
$MDM= -abs($ARGV[1]);
$MDM2= abs($ARGV[2]);
$MDP= 3*abs($ARGV[1]);


my $expnb = 0;



# Checking meta stable dark higgs

$longlived = 1;




 if(!open(FO,">$outname"))
 {
     exit;
   }


  print FO "
#Parameter Card for Various beam dump experiments
#All masses should be provided in GeV, all lengths in meters.
#Lines preceded by a # are ignored by the parser.

#Model Parameters
epsilon 1e-3
dark_matter_mass $MDM
dark_photon_mass $MDP
alpha_D 0.1
dark_matter2_mass $MDM2
dark_higgs_mass $MDP
dark_matter_Z11 0.7
dark_matter_Z12 -0.7
dark_matter_Z21 0.7
dark_matter_Z22 0.7



";



if( $expnb == 0 ){
##################################################################
######          For mBoone                                   #######
##################################################################
     print FO "
#Parameter Card for mBoone

#Run parameters
POT 1.87e20
pi0_per_POT 2.4
# We conserve the miniboone efficiency, this is very conservative ...
efficiency 1

samplesize 50

burn_max 50
burn_timeout 20000

beam_energy 8.89



################################
#Production Channel Definitions#
################################

";

 if ($longlived and $fmass < 0.03) {
print FO  "
# First production channel

production_channel pi0_decay

";
}

if ($longlived and $fmass > 0.03 and $fmass < 0.134976) {
print FO  "
# First production channel

production_channel pi0_decay

production_channel eta_decay
meson_per_pi0 0.033

";
}


if($longlived and $fmass<0.546 and $fmass > 0.134976)
{
     print FO  "

production_channel eta_decay
meson_per_pi0 0.033

production_channel omega_decay
meson_per_pi0 0.046

production_channel rho_decay
meson_per_pi0 0.05

production_channel V_decay
production_distribution proton_brem
ptmax 0.2
zmin 0.3
zmax 0.7

";
}


if($longlived and $fmass>0.546 )
{
     print FO  "

production_channel omega_decay
meson_per_pi0 0.046

production_channel rho_decay
meson_per_pi0 0.05

production_channel V_decay
production_distribution proton_brem
ptmax 0.2
zmin 0.3
zmax 0.7

";
}

    print FO "

############################
#END OF PRODUCTION CHANNELS#
############################


################
#SIGNAL CHANNEL#
################


signal_channel NCE_electron


########
#OUPTUT#
########

#Where to write events.
output_file Events/events.dat
#Where to write a summary of the run with number of events and paramaters in the format: channel_name V_mass DM_mass num_events epsilon alpha_prime scattering_channel
summary_file $BdNMCout


output_mode summary

#Cuts on the kinetic energy of outgoing nucleon or electron. These default to min=0 and max=1e9 GeV

max_scatter_energy 0.6
min_scatter_energy 0.050
min_scatter_angle 0.035
max_scatter_angle 7

######################
#DETECTOR DECLARATION#
######################

#Detector Parameters
detector sphere
x-position 0.0
y-position -1.9
z-position 491.0
radius 5.0

#Material parameters
#Mass is set in GeV.
#mass is only important for coherent scattering, can be set to anything.
#anything not defined will be set to zero.
material Carbon
number_density 3.63471e22
proton_number 6
neutron_number 6
electron_number 6
mass 11.2593

material Hydrogen
number_density 7.26942e22
proton_number 1
neutron_number 0
electron_number 1
mass 0.945778

 ";
 close F0;
}
else{
 print "Error, invalid experiment number $nbexp";
}
