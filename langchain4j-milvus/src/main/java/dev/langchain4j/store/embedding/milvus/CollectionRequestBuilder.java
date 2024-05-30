package dev.langchain4j.store.embedding.milvus;

import dev.langchain4j.store.embedding.filter.Filter;
import io.milvus.common.clientenum.ConsistencyLevelEnum;
import io.milvus.param.MetricType;
import io.milvus.param.collection.DropCollectionParam;
import io.milvus.param.collection.FlushParam;
import io.milvus.param.collection.HasCollectionParam;
import io.milvus.param.collection.LoadCollectionParam;
import io.milvus.param.dml.DeleteParam;
import io.milvus.param.collection.*;
import io.milvus.param.dml.InsertParam;
import io.milvus.param.dml.QueryParam;
import io.milvus.param.dml.SearchParam;
import io.milvus.param.highlevel.dml.DeleteIdsParam;
import io.milvus.param.index.DropIndexParam;
import io.milvus.param.partition.CreatePartitionParam;
import io.milvus.param.partition.DropPartitionParam;
import io.milvus.param.partition.HasPartitionParam;
import io.milvus.param.partition.ReleasePartitionsParam;
import org.apache.commons.collections4.CollectionUtils;
import org.apache.commons.lang3.StringUtils;

import java.util.Collections;
import java.util.List;
import java.util.Objects;

import static dev.langchain4j.store.embedding.milvus.MilvusEmbeddingStore.*;
import static java.lang.String.format;
import static java.util.Arrays.asList;
import static java.util.Collections.singletonList;
import static java.util.stream.Collectors.joining;

class CollectionRequestBuilder {

    static FlushParam buildFlushRequest(String collectionName) {
        return FlushParam.newBuilder()
                .withCollectionNames(singletonList(collectionName))
                .build();
    }

    static HasCollectionParam buildHasCollectionRequest(String collectionName) {
        return HasCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build();
    }

    static HasPartitionParam buildHasPartitionsRequest(String collectionName, String partitionName) {
        return HasPartitionParam.newBuilder()
                .withCollectionName(collectionName)
                .withPartitionName(partitionName)
                .build();
    }

    static DropCollectionParam buildDropCollectionRequest(String collectionName) {
        return DropCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build();
    }

    static CreatePartitionParam buildCreatePartitionRequest(String collectionName, String partitionName) {
        return CreatePartitionParam.newBuilder()
                .withCollectionName(collectionName)
                .withPartitionName(partitionName)
                .build();
    }

    static DropPartitionParam buildDropPartitionRequest(String collectionName, String partitionName) {
        return DropPartitionParam.newBuilder()
                .withCollectionName(collectionName)
                .withPartitionName(partitionName)
                .build();
    }


    static InsertParam buildInsertRequest(String collectionName, List<InsertParam.Field> fields) {
        return InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields)
                .build();
    }

    static InsertParam buildInsertRequest(String collectionName, List<InsertParam.Field> fields, String partitionName) {
        InsertParam.Builder builder = InsertParam.newBuilder()
                .withCollectionName(collectionName)
                .withFields(fields);
        if (StringUtils.isNotEmpty(partitionName)) {
            builder.withPartitionName(partitionName);
        }
        return builder.build();
    }

    static DropIndexParam buildDropIndexRequest(String collectionName, String indexName) {
        DropIndexParam.Builder builder = DropIndexParam.newBuilder()
                .withCollectionName(collectionName);
        if (StringUtils.isNotEmpty(indexName)) {
            builder.withIndexName(indexName);
        }
        return builder.build();
    }

    static DeleteIdsParam buildDeleteByIdsRequest(String collectionName, String partitionName, List<String> primaryIds) {
        DeleteIdsParam.Builder builder = DeleteIdsParam.newBuilder()
                .withCollectionName(collectionName)
                .withPrimaryIds(primaryIds);
        if (StringUtils.isNotEmpty(partitionName)) {
            builder.withPartitionName(partitionName);
        }
        return builder.build();
    }

    static LoadCollectionParam buildLoadCollectionInMemoryRequest(String collectionName) {
        return LoadCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build();
    }

    static GetLoadStateParam buildGetLoadStateRequest(String collectionName, List<String> partitionNames) {
        GetLoadStateParam.Builder requestBuilder =  GetLoadStateParam.newBuilder().withCollectionName(collectionName);
        if (CollectionUtils.isNotEmpty(partitionNames)) {
            requestBuilder.withPartitionNames(partitionNames);
        }
       return requestBuilder.build();
    }

    static ReleaseCollectionParam buildReleaseCollectionInMemoryRequest(String collectionName) {
        return ReleaseCollectionParam.newBuilder()
                .withCollectionName(collectionName)
                .build();
    }

    static ReleasePartitionsParam buildReleasePartitionsInMemoryRequest(String collectionName, List<String> partitionNames) {
        return ReleasePartitionsParam.newBuilder()
                .withCollectionName(collectionName)
                .withPartitionNames(partitionNames)
                .build();
    }

    static SearchParam buildSearchRequest(String collectionName,
                                          List<Float> vector,
                                          Filter filter,
                                          int maxResults,
                                          MetricType metricType,
                                          ConsistencyLevelEnum consistencyLevel) {
        return buildSearchRequest(collectionName, Collections.emptyList(), vector, filter, maxResults, metricType, consistencyLevel);
    }

    static SearchParam buildSearchRequest(String collectionName,
                                          List<String> partitionNames,
                                          List<Float> vector,
                                          Filter filter,
                                          int maxResults,
                                          MetricType metricType,
                                          ConsistencyLevelEnum consistencyLevel) {

        SearchParam.Builder builder = SearchParam.newBuilder()
                .withCollectionName(collectionName)
                .withVectors(singletonList(vector))
                .withVectorFieldName(VECTOR_FIELD_NAME)
                .withTopK(maxResults)
                .withMetricType(metricType)
                .withConsistencyLevel(consistencyLevel)
                .withOutFields(asList(ID_FIELD_NAME, TEXT_FIELD_NAME, METADATA_FIELD_NAME));

        if (filter != null) {
            builder.withExpr(MilvusMetadataFilterMapper.map(filter));
        }
        if (Objects.nonNull(partitionNames) && !partitionNames.isEmpty()) {
            builder.withPartitionNames(partitionNames);
        }

        return builder.build();
    }

    static QueryParam buildQueryRequest(String collectionName,
                                        List<String> rowIds,
                                        ConsistencyLevelEnum consistencyLevel) {
        return QueryParam.newBuilder()
                .withCollectionName(collectionName)
                .withExpr(buildQueryExpression(rowIds))
                .withConsistencyLevel(consistencyLevel)
                .withOutFields(singletonList(VECTOR_FIELD_NAME))
                .build();
    }

    static DeleteParam buildDeleteRequest(String collectionName,
                                          String expr) {
        return DeleteParam.newBuilder()
                .withCollectionName(collectionName)
                .withExpr(expr)
                .build();
    }

    private static String buildQueryExpression(List<String> rowIds) {
        return rowIds.stream()
                .map(id -> format("%s == '%s'", ID_FIELD_NAME, id))
                .collect(joining(" || "));
    }
}
