use rand::{distributions::Distribution, rngs::OsRng};
use sss::{
    artin::GF65536NPreparedMultipointEvalVZG,
    field::F2,
    galois::{GF65536NTower, GF65536N},
    goppa::{generate, BinaryGoppaDecoder, BinaryPacked, GoppaDecoder, GoppaEncoder},
};

#[test]
#[ignore]
fn encode_decode_large() {
    let n = 6960;
    let t = 94;
    let u = rand::distributions::uniform::Uniform::from(0..n);
    let (enc, dec) =
        generate::<GF65536N, GF65536NTower, GF65536NPreparedMultipointEvalVZG, _>(&mut OsRng, t, n);
    let dec = BinaryGoppaDecoder::from_decoder(dec);
    for _ in 0..2 {
        let mut x = vec![0; n];
        for _ in 0..t {
            x[u.sample(&mut OsRng)] = 1;
        }
        let x: Vec<_> = x.into_iter().map(F2).collect();
        let y = enc.encode(&x);
        let (_, error) = dec.decode(y);
        assert_eq!(error, x);
        eprintln!("done")
    }
}

#[test]
fn decode_large() {
    let n = 6960;
    let t = 94;
    let u = rand::distributions::uniform::Uniform::from(0..n);
    let BinaryPacked(enc): BinaryPacked<GoppaEncoder<F2, GF65536NTower>> =
        serde_json::from_str(include_str!("mc.pub")).unwrap();
    let BinaryPacked(dec): BinaryPacked<
        GoppaDecoder<GF65536N, GF65536NTower, GF65536NPreparedMultipointEvalVZG>,
    > = serde_json::from_str(include_str!("mc.pri")).unwrap();
    let dec = BinaryGoppaDecoder::from_decoder(dec);
    for _ in 0..128 {
        let mut x = vec![0; n];
        for _ in 0..t {
            x[u.sample(&mut OsRng)] = 1;
        }
        let x: Vec<_> = x.into_iter().map(F2).collect();
        let y = enc.encode(&x);
        let (_, error) = dec.decode(y);
        assert_eq!(error, x);
        eprintln!("done")
    }
}
