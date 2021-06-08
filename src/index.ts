import axios from "axios";
import Web3 from "web3";
import fs from "fs";
import path from "path";
import { exec } from "shelljs";
// @ts-ignore
import erc721 from "./erc721.json";

const provider = new Web3.providers.WebsocketProvider(
    "wss://wsapi.fantom.network",
    {
        reconnect: {
            auto: true,
        },
    }
);
const web3 = new Web3(provider);
const contract = new web3.eth.Contract(
    erc721.output.abi,
    "0x3d7071E5647251035271Aeb780d832B381Fa730F"
);

async function downloadImage(url: any, path: any) {
    const writer = fs.createWriteStream(`${path}/image.png`);

    const response = await axios({
        url,
        method: "GET",
        responseType: "stream",
    });

    response.data.pipe(writer);

    return new Promise((resolve, reject) => {
        writer.on("finish", resolve);
        writer.on("error", reject);
    });
}

const init = async () => {
    await exec(
        `find /Users/alsinas/Projects/gantom-backup/backup -size 0 -delete`,
        { async: true }
    );
    await exec(
        `find /Users/alsinas/Projects/gantom-backup/backup -type d -empty -delete`,
        { async: true }
    );

    const totalSupply = await contract.methods.totalSupply().call();
    console.log(`Total supply: ${totalSupply}`);

    for (var i = 0; i < totalSupply; i++) {
        try {
            const path = `./backup/${i}`;
            fs.mkdirSync(path);

            const tokenURI = await contract.methods.tokenURI(i).call();

            console.log(i, tokenURI);

            const { data } = await axios.get(tokenURI);

            await fs.writeFileSync(
                `${path}/metadata.json`,
                JSON.stringify(data, null, 4)
            );
            await fs.writeFileSync(
                `${path}/links.txt`,
                `${tokenURI}\r\n${data.image}`
            );
            await downloadImage(data.image, path);
        } catch (error) {
            // console.error(`An error occurred while backing up #${i}.`);
        }
    }
};

init();
