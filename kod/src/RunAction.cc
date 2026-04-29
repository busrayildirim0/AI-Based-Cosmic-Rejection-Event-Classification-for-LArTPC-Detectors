#include "RunAction.hh"

#include "G4Run.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4Version.hh"
#include "G4AutoLock.hh"

#include <ctime>
#include <iomanip>
#include <cmath>


namespace {
    G4Mutex runActionMutex = G4MUTEX_INITIALIZER;
}

RunAction::RunAction()
: G4UserRunAction(),
  fOutputFileName(""),
  fDataOutputInitialized(false)
{
    fEnergyDeposit = 0.;
    fTrackLength = 0.;
    fCosmicEvents = 0;
    fNeutrinoEvents = 0;
    fTestEvents = 0;
}

RunAction::~RunAction()
{
    if (fDataFile.is_open())
        fDataFile.close();
}

void RunAction::BeginOfRunAction(const G4Run*)
{
    G4AutoLock lock(&runActionMutex);
    
    fEnergyDeposit = 0.;
    fTrackLength = 0.;
    fCosmicEvents = 0;
    fNeutrinoEvents = 0;
    fTestEvents = 0;
    
    InitializeDataOutput();
    
    G4cout << "========================================" << G4endl;
    G4cout << "Run initialized successfully" << G4endl;
    G4cout << "Output file: " << fOutputFileName << G4endl;
    G4cout << "========================================" << G4endl;
}

void RunAction::EndOfRunAction(const G4Run* run)
{
    G4AutoLock lock(&runActionMutex);
    
    G4int nofEvents = run->GetNumberOfEvent();
    if (nofEvents == 0) return;

    if (fDataFile.is_open()) {
        fDataFile.close();
    }
    
    G4cout << "\n========================================" << G4endl;
    G4cout << "Run completed successfully" << G4endl;
    G4cout << "Total events processed: " << nofEvents << G4endl;
    G4cout << "  Cosmic events: " << fCosmicEvents.GetValue() << G4endl;
    G4cout << "  Neutrino events: " << fNeutrinoEvents.GetValue() << G4endl;
    G4cout << "  Test events: " << fTestEvents.GetValue() << G4endl;
    G4cout << "Total energy deposit: " 
           << fEnergyDeposit.GetValue()/GeV << " GeV" << G4endl;
    G4cout << "Total track length: " 
           << fTrackLength.GetValue()/m << " m" << G4endl;
    G4cout << "Output saved to: " << fOutputFileName << G4endl;
    G4cout << "========================================" << G4endl;
}


void RunAction::InitializeDataOutput()
{
    if (fDataOutputInitialized) return;

    time_t now = time(nullptr);
    tm* t = localtime(&now);
    char timestamp[64];
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", t);

    fOutputFileName = "lar_tpc_data_" + G4String(timestamp) + ".csv";
    fDataFile.open(fOutputFileName.c_str());

    if (!fDataFile.is_open()) {
        G4Exception(
            "RunAction::InitializeDataOutput",
            "IO001",
            FatalException,
            "Cannot open CSV output file."
        );
    }

    
    fDataFile << "# LAr TPC Simulation Dataset\n";
    fDataFile << "# Generated: " << timestamp << "\n";
    fDataFile << "# Geant4 Version: " << G4VERSION_NUMBER << "\n";
    fDataFile << "# Physics List: FTFP_BERT with EmStandard_option4\n";
    fDataFile << "# Detector: MicroBooNE-style LAr TPC (256x233x1037 cm^3)\n";
    fDataFile << "# Wire Spacing: 0.3 cm | Drift Velocity: 0.16 cm/microsecond\n";
    fDataFile << "# Features: 43 geometric/topological (reco-level)\n";
    fDataFile << "#\n";

    fDataFile << std::fixed << std::setprecision(6);

    fDataFile 
        << "EventID,"
        << "EventType,"
        << "InteractionType,"
        
        << "Enu_GeV,"
        << "Q2_GeV2,"
        << "W_GeV,"
        
        << "vertex_x_cm,"
        << "vertex_y_cm,"
        << "vertex_z_cm,"
        
        << "primary_px_GeV,"
        << "primary_py_GeV,"
        << "primary_pz_GeV,"
        << "primary_p_GeV,"
        << "primary_E_GeV,"
        << "primary_theta_deg,"
        << "primary_phi_deg,"
        << "primary_PDG,"
        
        << "nHits,"
        << "VisibleEnergy_MeV,"
        << "AvgHitEnergy_MeV,"
        
        << "hit_length_x_cm,"
        << "hit_length_y_cm,"
        << "hit_length_z_cm,"
        << "hit_aspect_ratio,"
        
        << "pca_eigenvalue_1,"
        << "pca_eigenvalue_2,"
        << "pca_eigenvalue_3,"
        << "pca_ratio_12,"
        << "pca_ratio_13,"
        
        << "opening_angle_deg,"
        << "track_angle_wrt_beam_deg,"
        
        << "dEdx_mean_MeV_cm,"
        << "dEdx_std_MeV_cm,"
        << "energy_front_fraction,"
        << "energy_back_fraction,"
        
        << "hit_density_per_cm3,"
        << "hit_time_spread_us,"
        << "n_isolated_hits,"
        
        << "nHits_U,"
        << "nHits_V,"
        << "nHits_Y,"
        << "hit_plane_ratio_Y_UV"
        << "\n";

    fDataFile.flush();
    fDataOutputInitialized = true;
}

void RunAction::WriteEventData(
    G4int eventID,
    G4int eventType,
    const G4String& interactionType,
    



    G4double neutrinoEnu,
    G4double neutrinoQ2,
    G4double neutrinoW,
    
    G4double vertexX,
    G4double vertexY,
    G4double vertexZ,
    
    G4double primaryPx,
    G4double primaryPy,
    G4double primaryPz,
    G4double primaryP,
    G4double primaryE,
    G4double primaryTheta,
    G4double primaryPhi,
    G4int primaryPDG,
    
    G4int nHits,
    G4double visibleEnergy,
    G4double avgHitEnergy,
    
    G4double hitLengthX,
    G4double hitLengthY,
    G4double hitLengthZ,
    G4double hitAspectRatio,
    
    G4double pca1,
    G4double pca2,
    G4double pca3,
    G4double pcaRatio12,
    G4double pcaRatio13,
    
    G4double openingAngle,
    G4double trackAngleWrtBeam,
    
    G4double dEdxMean,
    G4double dEdxStd,
    G4double energyFrontFraction,
    G4double energyBackFraction,
    
    G4double hitDensity,
    G4double hitTimeSpread,
    G4int nIsolatedHits,
    
    G4int nHitsU,
    G4int nHitsV,
    G4int nHitsY,
    G4double hitPlaneRatioYUV
)
{
    G4AutoLock lock(&runActionMutex);
    
    if (!fDataFile.is_open()) {
        G4cerr << "ERROR: Data file not open in WriteEventData!" << G4endl;
        return;
    }

    auto safeNum = [](G4double val) -> G4double {
        return (std::isfinite(val)) ? val : -999.0;
    };

    fDataFile 
        
        << eventID << ","
        << eventType << ","
        << interactionType << ","
        
        << safeNum(neutrinoEnu) << ","
        << safeNum(neutrinoQ2) << ","
        << safeNum(neutrinoW) << ","
        
        << safeNum(vertexX) << ","
        << safeNum(vertexY) << ","
        << safeNum(vertexZ) << ","
        
        << safeNum(primaryPx) << ","
        << safeNum(primaryPy) << ","
        << safeNum(primaryPz) << ","
        << safeNum(primaryP) << ","
        << safeNum(primaryE) << ","
        << safeNum(primaryTheta) << ","
        << safeNum(primaryPhi) << ","
        << primaryPDG << ","
        
        << nHits << ","
        << safeNum(visibleEnergy) << ","
        << safeNum(avgHitEnergy) << ","
        
        << safeNum(hitLengthX) << ","
        << safeNum(hitLengthY) << ","
        << safeNum(hitLengthZ) << ","
        << safeNum(hitAspectRatio) << ","
        
        << safeNum(pca1) << ","
        << safeNum(pca2) << ","
        << safeNum(pca3) << ","
        << safeNum(pcaRatio12) << ","
        << safeNum(pcaRatio13) << ","
        
        << safeNum(openingAngle) << ","
        << safeNum(trackAngleWrtBeam) << ","
        
        << safeNum(dEdxMean) << ","
        << safeNum(dEdxStd) << ","
        << safeNum(energyFrontFraction) << ","
        << safeNum(energyBackFraction) << ","
        
        << safeNum(hitDensity) << ","
        << safeNum(hitTimeSpread) << ","
        << nIsolatedHits << ","
        
        << nHitsU << ","
        << nHitsV << ","
        << nHitsY << ","
        << safeNum(hitPlaneRatioYUV)
        << "\n";

    fDataFile.flush();
}

void RunAction::AddEnergyDeposit(G4double edep)
{
    fEnergyDeposit += edep;
}

void RunAction::AddTrackLength(G4double length)
{
    fTrackLength += length;
}

void RunAction::IncrementEventType(G4int eventType)
{
    if (eventType == 0) {
        fCosmicEvents += 1;
    } else if (eventType == 1) {
        fNeutrinoEvents += 1;
    } else {
        fTestEvents += 1;
    }
}

void RunAction::FinalizeDataOutput()
{
    G4AutoLock lock(&runActionMutex);
    if (fDataFile.is_open()) {
        fDataFile.close();
    }
}
