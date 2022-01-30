use std::{borrow::Borrow, hash::Hash, marker::PhantomData};

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub struct MerkleTreeNode<T, S, H, Ho> {
    left: Box<MerkleTree<T, S, H, Ho>>,
    right: Box<MerkleTree<T, S, H, Ho>>,
    depth: usize,
    hash: Ho,
}

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub struct MerkleTreeLeaf<T, S, H, Ho> {
    data: T,
    proof: S,
    hash: Option<Ho>,
    _p: PhantomData<fn() -> H>,
}

#[derive(Clone, Debug, Hash, PartialEq, PartialOrd, Ord, Eq)]
pub enum MerkleTree<T, S, H, Ho> {
    Node(MerkleTreeNode<T, S, H, Ho>),
    Leaf(MerkleTreeLeaf<T, S, H, Ho>),
}

impl<T, S, H, Ho> MerkleTree<T, S, H, Ho> {
    fn make_leaf_hash(&mut self, hasher: &H)
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        match self {
            Self::Leaf(MerkleTreeLeaf {
                data, proof, hash, ..
            }) => {
                let data = <H as MerkleHash<T>>::hash(hasher, data);
                let proof = <H as MerkleHash<S>>::hash(hasher, proof);
                *hash = Some(hasher.hash((data, proof)));
            }
            _ => panic!("logical error: expecting a leaf"),
        }
    }

    fn new_node(mut left: Self, mut right: Self, hasher: &H) -> Self
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        let depth = match (&left, &right) {
            (
                Self::Node(MerkleTreeNode {
                    depth: depth_left, ..
                }),
                Self::Node(MerkleTreeNode {
                    depth: depth_right, ..
                }),
            ) => {
                assert_eq!(depth_left, depth_right);
                depth_left + 1
            }
            (Self::Leaf { .. }, Self::Leaf { .. }) => 1,
            _ => panic!("logical error: depth mismatch"),
        };
        if let Self::Leaf(MerkleTreeLeaf { hash: None, .. }) = left {
            left.make_leaf_hash(hasher)
        }
        if let Self::Leaf(MerkleTreeLeaf { hash: None, .. }) = right {
            right.make_leaf_hash(hasher)
        }
        let hash = match (&mut left, &mut right) {
            (
                Self::Leaf(MerkleTreeLeaf {
                    hash: hash_left, ..
                }),
                Self::Leaf(MerkleTreeLeaf {
                    hash: hash_right, ..
                }),
            ) => {
                let hash_left = hash_left.as_ref().unwrap().clone();
                let hash_right = hash_right.as_ref().unwrap().clone();
                hasher.hash((hash_left, hash_right))
            }
            (
                Self::Node(MerkleTreeNode {
                    hash: hash_left, ..
                }),
                Self::Node(MerkleTreeNode {
                    hash: hash_right, ..
                }),
            ) => {
                let hash_left = hash_left.clone();
                let hash_right = hash_right.clone();
                hasher.hash((hash_left, hash_right))
            }
            _ => panic!("unexpected left right tree nodes"),
        };
        let left = Box::new(left);
        let right = Box::new(right);
        Self::Node(MerkleTreeNode {
            left,
            right,
            depth,
            hash,
        })
    }

    pub fn new_leaf(data: T, proof: S) -> Self {
        Self::Leaf(MerkleTreeLeaf {
            data,
            proof,
            hash: None,
            _p: PhantomData,
        })
    }

    pub fn build_tree(leaves: Vec<Self>, hasher: &H) -> Self
    where
        H: MerkleHash<T, Output = Ho>
            + MerkleHash<S, Output = Ho>
            + MerkleHash<(Ho, Ho), Output = Ho>,
        Ho: Eq + Hash + Clone,
    {
        let mut len = leaves.len();
        assert!(len > 1 && len.is_power_of_two());
        assert!(leaves.iter().all(|node| match node {
            Self::Leaf { .. } => true,
            _ => false,
        }));
        let mut nodes = leaves;
        while nodes.len() > 1 {
            let mut nodes_itr = nodes.into_iter();
            let mut new_nodes = vec![];
            len >>= 1;
            for _ in 0..len {
                let left = nodes_itr
                    .next()
                    .expect("number of leaves should be power of 2");
                let right = nodes_itr
                    .next()
                    .expect("number of leaves should be power of 2");
                new_nodes.push(Self::new_node(left, right, hasher));
            }
            nodes = new_nodes;
        }
        nodes.pop().unwrap()
    }

    pub fn iter(&self) -> TreeIter<T, S, H, Ho> {
        TreeIter::new(self)
    }
}

pub struct TreeIter<'a, T, S, H, Ho> {
    path: Vec<(&'a MerkleTree<T, S, H, Ho>, Vec<(Ho, MerkleTreeNodeType)>)>,
}

impl<'a, T, S, H, Ho> TreeIter<'a, T, S, H, Ho> {
    fn new(root: &'a MerkleTree<T, S, H, Ho>) -> Self {
        Self {
            path: vec![(root, vec![])],
        }
    }
}

impl<'a, T, S, H, Ho> Iterator for TreeIter<'a, T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash + Clone,
{
    type Item = MerkleTreePath<'a, 'a, T, S, H, Ho>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some((node, mut path)) = self.path.pop() {
                match node {
                    MerkleTree::Leaf(MerkleTreeLeaf { data, proof, .. }) => {
                        path.reverse();
                        break Some(MerkleTreePath {
                            data,
                            proof,
                            path,
                            _p: PhantomData,
                        });
                    }
                    MerkleTree::Node(MerkleTreeNode { left, right, .. }) => {
                        self.path.push((right, {
                            let mut path = path.clone();
                            let sibling_hash = match left as &MerkleTree<_, _, _, Ho> {
                                MerkleTree::Node(MerkleTreeNode { hash, .. }) => hash.clone(),
                                MerkleTree::Leaf(MerkleTreeLeaf { hash, .. }) => {
                                    hash.as_ref().expect("hash should be populated").clone()
                                }
                            };
                            path.push((sibling_hash, MerkleTreeNodeType::Left));
                            path
                        }));
                        self.path.push((left, {
                            let mut path = path.clone();
                            let sibling_hash = match right as &MerkleTree<_, _, _, Ho> {
                                MerkleTree::Node(MerkleTreeNode { hash, .. }) => hash.clone(),
                                MerkleTree::Leaf(MerkleTreeLeaf { hash, .. }) => {
                                    hash.as_ref().expect("hash should be populated").clone()
                                }
                            };
                            path.push((sibling_hash, MerkleTreeNodeType::Right));
                            path
                        }));
                    }
                }
            } else {
                break None;
            }
        }
    }
}

pub trait MerkleHash<T> {
    type Output: Eq + Hash;
    fn hash<S: Borrow<T>>(&self, data: S) -> Self::Output;
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub enum MerkleTreeNodeType {
    Left,
    Right,
}

#[derive(Debug)]
pub struct MerkleTreePath<'a, 'b, T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash,
{
    pub data: &'a T,
    pub proof: &'b S,
    path: Vec<(Ho, MerkleTreeNodeType)>,
    _p: PhantomData<fn() -> H>,
}

impl<'a, 'b, T, S, H, Ho> MerkleTreePath<'a, 'b, T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho> + MerkleHash<(Ho, Ho), Output = Ho>,
    Ho: Eq + Hash,
{
    pub fn new(data: &'a T, proof: &'b S, path: Vec<(Ho, MerkleTreeNodeType)>) -> Self {
        Self {
            data,
            proof,
            path,
            _p: PhantomData,
        }
    }

    pub fn root_hash(&self, hasher: &H) -> Ho
    where
        Ho: Clone,
    {
        let tuple: (Ho, Ho) = (
            <H as MerkleHash<T>>::hash(&hasher, self.data),
            <H as MerkleHash<S>>::hash(&hasher, self.proof),
        );
        let mut hash = hasher.hash(tuple);
        for (sibling_hash, sibling_type) in &self.path {
            let tuple = match sibling_type {
                MerkleTreeNodeType::Left => (hash, sibling_hash.clone()),
                MerkleTreeNodeType::Right => (sibling_hash.clone(), hash),
            };
            hash = hasher.hash(tuple);
        }
        hash
    }

    pub fn lemmas(&self) -> &[(Ho, MerkleTreeNodeType)] {
        &self.path
    }
}

#[derive(Debug)]
pub struct MerkleTreeQuorumLeaf<Ho> {
    hash: Ho,
}

#[derive(Debug)]
pub struct MerkleTreeQuorumNode<T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash,
{
    hash: Ho,
    left: Box<MerkleTreeQuorum<T, S, H, Ho>>,
    right: Box<MerkleTreeQuorum<T, S, H, Ho>>,
    depth: usize,
}

#[derive(Debug)]
pub struct MerkleTreeQuorumUndetermined<T, S, H, Ho> {
    state: Option<(usize, Ho)>,
    _p: PhantomData<fn() -> (T, S, H)>,
}

#[derive(Debug)]
pub enum MerkleTreeQuorum<T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho>,
    Ho: Eq + Hash,
{
    Leaf(MerkleTreeQuorumLeaf<Ho>),
    Node(MerkleTreeQuorumNode<T, S, H, Ho>),
    Undetermined(MerkleTreeQuorumUndetermined<T, S, H, Ho>),
}

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("mismatch hash")]
    Mismatch,
}

impl<T, S, H, Ho> MerkleTreeQuorum<T, S, H, Ho>
where
    H: MerkleHash<T, Output = Ho> + MerkleHash<S, Output = Ho> + MerkleHash<(Ho, Ho), Output = Ho>,
    Ho: Eq + Hash + Clone,
    T: Eq,
    S: Eq,
{
    pub fn new() -> Self {
        Self::Undetermined(MerkleTreeQuorumUndetermined {
            state: None,
            _p: PhantomData,
        })
    }

    fn leaf(data: &T, proof: &S, hasher: &H) -> Self {
        let tuple: (Ho, Ho) = (
            <H as MerkleHash<T>>::hash(&hasher, data),
            <H as MerkleHash<S>>::hash(&hasher, proof),
        );
        let hash = hasher.hash(tuple);
        Self::Leaf(MerkleTreeQuorumLeaf { hash })
    }

    fn try_admit_rec(
        &mut self,
        hasher: &H,
        path: MerkleTreePath<T, S, H, Ho>,
    ) -> Result<(), Error> {
        let MerkleTreePath {
            data,
            proof,
            mut path,
            ..
        } = path;
        match self {
            Self::Undetermined(MerkleTreeQuorumUndetermined {
                state: Some((depth, hash)),
                ..
            }) if *depth == path.len() => {
                *self = match path.pop() {
                    None => Self::leaf(data, proof, hasher),
                    Some((hash_sibling, sibling_type)) => {
                        let depth = path.len() + 1;
                        let mut node = Self::new();
                        node.try_admit_rec(
                            hasher,
                            MerkleTreePath {
                                data,
                                proof,
                                path,
                                _p: PhantomData,
                            },
                        )?;
                        let hash_child = match &node {
                            Self::Node(MerkleTreeQuorumNode {
                                hash: hash_child, ..
                            }) => hash_child.clone(),
                            Self::Leaf(MerkleTreeQuorumLeaf {
                                hash: hash_child, ..
                            }) => hash_child.clone(),
                            _ => return Err(Error::Mismatch),
                        };

                        let hash_expected = {
                            let tuple = match sibling_type {
                                MerkleTreeNodeType::Left => {
                                    (hash_sibling.clone(), hash_child.clone())
                                }
                                MerkleTreeNodeType::Right => {
                                    (hash_child.clone(), hash_sibling.clone())
                                }
                            };
                            hasher.hash(tuple)
                        };
                        if *hash != hash_expected {
                            return Err(Error::Mismatch);
                        }
                        let sibling = Box::new(Self::Undetermined(MerkleTreeQuorumUndetermined {
                            state: Some((depth - 1, hash_sibling)),
                            _p: PhantomData,
                        }));
                        match sibling_type {
                            MerkleTreeNodeType::Left => Self::Node(MerkleTreeQuorumNode {
                                hash: hash.clone(),
                                depth,
                                right: Box::new(node),
                                left: sibling,
                            }),
                            MerkleTreeNodeType::Right => Self::Node(MerkleTreeQuorumNode {
                                hash: hash.clone(),
                                depth,
                                left: Box::new(node),
                                right: sibling,
                            }),
                        }
                    }
                };
                Ok(())
            }
            Self::Undetermined(MerkleTreeQuorumUndetermined { state: None, .. }) => {
                *self = match path.pop() {
                    None => Self::leaf(data, proof, hasher),
                    Some((hash_sibling, sibling_type)) => {
                        let depth = path.len() + 1;
                        let mut node = Self::new();
                        node.try_admit_rec(
                            hasher,
                            MerkleTreePath {
                                data,
                                proof,
                                path,
                                _p: PhantomData,
                            },
                        )?;
                        let hash_child = match &node {
                            Self::Node(MerkleTreeQuorumNode {
                                hash: hash_child, ..
                            }) => hash_child.clone(),
                            Self::Leaf(MerkleTreeQuorumLeaf {
                                hash: hash_child, ..
                            }) => hash_child.clone(),
                            _ => return Err(Error::Mismatch),
                        };

                        let hash = {
                            let tuple = match sibling_type {
                                MerkleTreeNodeType::Left => {
                                    (hash_sibling.clone(), hash_child.clone())
                                }
                                MerkleTreeNodeType::Right => {
                                    (hash_child.clone(), hash_sibling.clone())
                                }
                            };
                            hasher.hash(tuple)
                        };
                        let sibling = Box::new(Self::Undetermined(MerkleTreeQuorumUndetermined {
                            state: Some((depth - 1, hash_sibling)),
                            _p: PhantomData,
                        }));
                        match sibling_type {
                            MerkleTreeNodeType::Left => Self::Node(MerkleTreeQuorumNode {
                                hash: hash.clone(),
                                depth,
                                right: Box::new(node),
                                left: sibling,
                            }),
                            MerkleTreeNodeType::Right => Self::Node(MerkleTreeQuorumNode {
                                hash: hash.clone(),
                                depth,
                                left: Box::new(node),
                                right: sibling,
                            }),
                        }
                    }
                };
                Ok(())
            }
            Self::Leaf(MerkleTreeQuorumLeaf { hash: hash_quorom }) => {
                let tuple: (Ho, Ho) = (
                    <H as MerkleHash<T>>::hash(&hasher, data),
                    <H as MerkleHash<S>>::hash(&hasher, proof),
                );
                let hash = hasher.hash(tuple);
                if *hash_quorom == hash && path.is_empty() {
                    Ok(())
                } else {
                    Err(Error::Mismatch)
                }
            }
            Self::Node(MerkleTreeQuorumNode {
                left, right, depth, ..
            }) => match path.pop() {
                None => Err(Error::Mismatch),
                Some((hash_sibling, node_type)) => {
                    let (child, sibling) = match node_type {
                        MerkleTreeNodeType::Left => (&mut *right, &mut *left),
                        MerkleTreeNodeType::Right => (&mut *left, &mut *right),
                    };
                    match sibling.as_ref() {
                        Self::Leaf(MerkleTreeQuorumLeaf {
                            hash: sibling_hash_expected,
                            ..
                        }) if *depth == 1 && *sibling_hash_expected == hash_sibling => child
                            .try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            ),
                        Self::Node(MerkleTreeQuorumNode {
                            depth: sibling_depth,
                            hash: sibling_hash_expected,
                            ..
                        }) if sibling_depth + 1 == *depth
                            && *sibling_hash_expected == hash_sibling =>
                        {
                            child.try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            )
                        }
                        Self::Undetermined(MerkleTreeQuorumUndetermined {
                            state: Some((sibling_depth, sibling_hash_expected)),
                            ..
                        }) if sibling_depth + 1 == *depth
                            && *sibling_hash_expected == hash_sibling =>
                        {
                            child.try_admit_rec(
                                hasher,
                                MerkleTreePath {
                                    data,
                                    proof,
                                    path,
                                    _p: PhantomData,
                                },
                            )
                        }
                        _ => Err(Error::Mismatch),
                    }
                }
            },
            _ => Err(Error::Mismatch),
        }
    }

    pub fn try_admit(
        &mut self,
        hasher: &H,
        path: MerkleTreePath<T, S, H, Ho>,
    ) -> Result<(), Error> {
        self.try_admit_rec(hasher, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use sha2::{digest::Digest, Sha256};

    #[derive(Debug)]
    struct MyHasher(u64);

    impl MerkleHash<u8> for MyHasher {
        type Output = Vec<u8>;
        fn hash<S: Borrow<u8>>(&self, data: S) -> Self::Output {
            let mut d = Sha256::default();
            d.update(self.0.to_le_bytes());
            d.update(data.borrow().to_le_bytes());
            d.finalize().into_iter().collect()
        }
    }
    impl MerkleHash<(Vec<u8>, Vec<u8>)> for MyHasher {
        type Output = Vec<u8>;
        fn hash<S: Borrow<(Vec<u8>, Vec<u8>)>>(&self, data: S) -> Self::Output {
            let mut d = Sha256::default();
            d.update(self.0.to_le_bytes());
            let (a, b) = data.borrow();
            d.update(a);
            d.update(b", with another ");
            d.update(b);
            d.finalize().into_iter().collect()
        }
    }

    #[test]
    fn it_works() {
        let hasher = MyHasher(202);
        let leaves: Vec<_> = (0..16)
            .map(|idx| MerkleTree::new_leaf(idx, idx + 5))
            .collect();
        let tree = MerkleTree::build_tree(leaves, &hasher);
        let mut quorum = MerkleTreeQuorum::new();
        for path in tree.iter() {
            println!("{:?}", path);
            quorum.try_admit(&hasher, path).unwrap();
            println!("quorum={:?}", quorum);
        }
    }
}
