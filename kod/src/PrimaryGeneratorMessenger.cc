// PrimaryGeneratorMessenger.cc
#include "PrimaryGeneratorMessenger.hh"
#include "PrimaryGeneratorAction.hh"

#include "G4UIdirectory.hh"
#include "G4UIcommand.hh"
#include "G4UIparameter.hh"
#include "G4UIcmdWithAString.hh"
#include "G4UIcmdWithADoubleAndUnit.hh"
#include "G4UIcmdWith3VectorAndUnit.hh"
#include "G4SystemOfUnits.hh"

PrimaryGeneratorMessenger::PrimaryGeneratorMessenger(PrimaryGeneratorAction* genAction)
 : G4UImessenger(),
   fPrimaryGenerator(genAction),
   fGeneratorDir(nullptr),
   fSetModeCmd(nullptr),
   fSetNeutrinoEnergyCmd(nullptr),
   fSetCosmicEnergyCmd(nullptr),
   fSetCosmicAngleCmd(nullptr)
{
    fGeneratorDir = new G4UIdirectory("/generator/");
    fGeneratorDir->SetGuidance("Primary generator control");
    
    fSetModeCmd = new G4UIcmdWithAString("/generator/setMode", this);
    fSetModeCmd->SetGuidance("Set generator mode: neutrino, cosmic, or test");
    fSetModeCmd->SetParameterName("mode", false);
    fSetModeCmd->SetCandidates("neutrino cosmic test");
    fSetModeCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fSetNeutrinoEnergyCmd = new G4UIcommand("/generator/setNeutrinoEnergyRange", this);
    fSetNeutrinoEnergyCmd->SetGuidance("Set neutrino energy range (min max unit)");
    
    G4UIparameter* minEnuParam = new G4UIparameter("minE", 'd', false);
    minEnuParam->SetDefaultValue(0.2);
    fSetNeutrinoEnergyCmd->SetParameter(minEnuParam);
    
    G4UIparameter* maxEnuParam = new G4UIparameter("maxE", 'd', false);
    maxEnuParam->SetDefaultValue(3.0);
    fSetNeutrinoEnergyCmd->SetParameter(maxEnuParam);
    
    G4UIparameter* energyUnitParam = new G4UIparameter("unit", 's', false);
    energyUnitParam->SetDefaultValue("GeV");
    fSetNeutrinoEnergyCmd->SetParameter(energyUnitParam);
    
    fSetNeutrinoEnergyCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fSetCosmicEnergyCmd = new G4UIcommand("/generator/setCosmicEnergyRange", this);
    fSetCosmicEnergyCmd->SetGuidance("Set cosmic ray energy range (min max unit)");
    
    G4UIparameter* minCosmicEParam = new G4UIparameter("minE", 'd', false);
    minCosmicEParam->SetDefaultValue(0.1);
    fSetCosmicEnergyCmd->SetParameter(minCosmicEParam);
    
    G4UIparameter* maxCosmicEParam = new G4UIparameter("maxE", 'd', false);
    maxCosmicEParam->SetDefaultValue(100.0);
    fSetCosmicEnergyCmd->SetParameter(maxCosmicEParam);
    
    G4UIparameter* cosmicUnitParam = new G4UIparameter("unit", 's', false);
    cosmicUnitParam->SetDefaultValue("GeV");
    fSetCosmicEnergyCmd->SetParameter(cosmicUnitParam);
    
    fSetCosmicEnergyCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
    
    fSetCosmicAngleCmd = new G4UIcommand("/generator/setCosmicAngleRange", this);
    fSetCosmicAngleCmd->SetGuidance("Set cosmic ray angle range from zenith (min max unit)");
    
    G4UIparameter* minAngleParam = new G4UIparameter("minTheta", 'd', false);
    minAngleParam->SetDefaultValue(0.0);
    fSetCosmicAngleCmd->SetParameter(minAngleParam);
    
    G4UIparameter* maxAngleParam = new G4UIparameter("maxTheta", 'd', false);
    maxAngleParam->SetDefaultValue(85.0);
    fSetCosmicAngleCmd->SetParameter(maxAngleParam);
    
    G4UIparameter* angleUnitParam = new G4UIparameter("unit", 's', false);
    angleUnitParam->SetDefaultValue("deg");
    fSetCosmicAngleCmd->SetParameter(angleUnitParam);
    
    fSetCosmicAngleCmd->AvailableForStates(G4State_PreInit, G4State_Idle);
}

PrimaryGeneratorMessenger::~PrimaryGeneratorMessenger()
{
    delete fSetModeCmd;
    delete fSetNeutrinoEnergyCmd;
    delete fSetCosmicEnergyCmd;
    delete fSetCosmicAngleCmd;
    delete fGeneratorDir;
}

void PrimaryGeneratorMessenger::SetNewValue(G4UIcommand* command, G4String newValue)
{
    if (command == fSetModeCmd) {
        if (newValue == "neutrino") {
            fPrimaryGenerator->SetGeneratorMode(PrimaryGeneratorAction::kNeutrinoMode);
            G4cout << "Generator mode set to: NEUTRINO" << G4endl;
        }
        else if (newValue == "cosmic") {
            fPrimaryGenerator->SetGeneratorMode(PrimaryGeneratorAction::kCosmicRayMode);
            G4cout << "Generator mode set to: COSMIC RAY" << G4endl;
        }
        else if (newValue == "test") {
            fPrimaryGenerator->SetGeneratorMode(PrimaryGeneratorAction::kTestMode);
            G4cout << "Generator mode set to: TEST" << G4endl;
        }
    }
    else if (command == fSetNeutrinoEnergyCmd) {
        G4double minE, maxE;
        G4String unit;
        std::istringstream is(newValue);
        is >> minE >> maxE >> unit;
        
        G4double unitValue = G4UIcommand::ValueOf(unit.c_str());
        fPrimaryGenerator->SetNeutrinoEnergyRange(minE * unitValue, maxE * unitValue);
        
        G4cout << "Neutrino energy range set to: " 
               << minE << " - " << maxE << " " << unit << G4endl;
    }
    else if (command == fSetCosmicEnergyCmd) {
        G4double minE, maxE;
        G4String unit;
        std::istringstream is(newValue);
        is >> minE >> maxE >> unit;
        
        G4double unitValue = G4UIcommand::ValueOf(unit.c_str());
        fPrimaryGenerator->SetCosmicEnergyRange(minE * unitValue, maxE * unitValue);
        
        G4cout << "Cosmic energy range set to: " 
               << minE << " - " << maxE << " " << unit << G4endl;
    }
    else if (command == fSetCosmicAngleCmd) {
        G4double minTheta, maxTheta;
        G4String unit;
        std::istringstream is(newValue);
        is >> minTheta >> maxTheta >> unit;
        
        G4double unitValue = G4UIcommand::ValueOf(unit.c_str());
        fPrimaryGenerator->SetCosmicAngleRange(minTheta * unitValue, maxTheta * unitValue);
        
        G4cout << "Cosmic angle range set to: " 
               << minTheta << " - " << maxTheta << " " << unit << G4endl;
    }
}
