import { ethers } from "hardhat";

async function main() {
  console.log("Deploying GenomeVault VerificationContract...");

  // Get the contract factory
  const VerificationContract = await ethers.getContractFactory("VerificationContract");
  
  // Deploy the contract
  const verificationContract = await VerificationContract.deploy();
  
  // Wait for deployment to finish
  await verificationContract.waitForDeployment();
  
  const contractAddress = await verificationContract.getAddress();
  
  console.log("VerificationContract deployed to:", contractAddress);
  console.log("Contract owner:", await verificationContract.owner());
  
  // Display initial allowed circuits
  console.log("\nInitial allowed circuits:");
  const circuits = [
    "variant_presence",
    "polygenic_risk_score",
    "ancestry_composition",
    "pharmacogenomic",
    "pathway_enrichment",
    "diabetes_risk_alert"
  ];
  
  for (const circuit of circuits) {
    const allowed = await verificationContract.allowedCircuits(circuit);
    console.log(`  ${circuit}: ${allowed}`);
  }
  
  console.log("\nDeployment complete!");
  
  // Save deployment info
  const deploymentInfo = {
    contractAddress,
    network: "localhost",
    deployedAt: new Date().toISOString(),
    owner: await verificationContract.owner()
  };
  
  console.log("\nDeployment info:", JSON.stringify(deploymentInfo, null, 2));
}

// Execute deployment
main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
